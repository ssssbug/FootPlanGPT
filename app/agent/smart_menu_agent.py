import json
import os
import uuid
import hashlib
from datetime import datetime
from typing import Tuple, Optional, List

import jieba
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from agent.agent import Agent
from agent.baseAgent import BaseAgent
from llm.select_llm import LLM
from memory.WorkingMemory import WorkingMemory
from memory.baseMemory import MemoryItem, MemoryConfig
from memory.Semantic import SemanticMemory, Entity, Relation # Import Entity/Relation
from message.message import Message
from prompt.default_prompt import DEFAULT_REACT_TEMPLATE, INTENT_PROMPT, MEMORY_EXTRACTION_PROMPT # Added Prompt
from schemas.AgentState import AgentState
from utils.text_process import SessionUserId
from utils.text_process import TextProcess
#环境加载，加载llm
load_dotenv()

def generate_deterministic_id(name: str, type_str: str) -> str:
    """生成确定性ID: type_prefix + md5(name)"""
    # 规范化名称：去除空白，统一小写
    normalized_name = name.strip().lower()
    
    # 特殊处理：如果是当前用户，固定ID
    if normalized_name in ["current user", "me", "myself", "user", "用户"]:
         return "USER_CURRENT"
        
    # 计算MD5
    hash_object = hashlib.md5(normalized_name.encode())
    hash_hex = hash_object.hexdigest()[:8] # 取前8位即可，足够短且唯一性尚可
    
    # 构建ID
    # 如: INGREDIENT_a1b2c3d4
    return f"{type_str.upper()}_{hash_hex}"

##生成菜单的agent
class SmartMenuAgent(BaseAgent):
    """使用React模式的agent"""
    def __init__(self,
                 llm:Agent,
                 system_prompt:Optional[str]=None,
                 config:Optional[str]=None,
                 max_steps:int=20,
                 custom_prompt:Optional[str]=None
                 ):
        super().__init__(name="",llm=llm,system_prompt=system_prompt,Config=config)
        self.max_steps = max_steps
        self.current_history:List[str]=[]
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_TEMPLATE
        self.sessionuserid = SessionUserId()
        
        # 初始化语义记忆
        self.semantic_memory = SemanticMemory(MemoryConfig())

    def learn_from_interaction(self, user_input: str, agent_response: str):
        """
        从交互中学习新的实体和关系，并存入语义记忆
        """
        if not self.semantic_memory.graph_store:
            return

        print("\n[Self-Reflection] 正在提取对话中的新知识...")
        try:
            # 1. 构建提取 Prompt
            prompt = MEMORY_EXTRACTION_PROMPT.format(
                user_input=user_input,
                agent_response=agent_response
            )
            
            # 2. 调用 LLM
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            
            # 3. 解析结果
            try:
                # 尝试清洗可能存在的 Markdown 标记
                json_str = response.strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:-3]
                elif json_str.startswith("```"):
                    json_str = json_str[3:-3]
                
                data = json.loads(json_str)
                
                entities = data.get("entities", [])
                relations = data.get("relations", [])
                
                # ID 映射表: raw_id_from_llm -> deterministic_id
                id_map = {}
                
                count_e = 0
                count_r = 0
                
                # 4. 存入图数据库 - 实体
                for e_data in entities:
                    raw_id = e_data["id"]
                    name = e_data["name"]
                    e_type = e_data["type"]
                    
                    # 生成确定性ID
                    deterministic_id = generate_deterministic_id(name, e_type)
                    id_map[raw_id] = deterministic_id # 记录映射供 relations 使用
                    
                    entity = Entity(
                        entity_id=deterministic_id,
                        name=name,
                        entity_type=e_type,
                        properties=e_data.get("properties")
                    )
                    if self.semantic_memory.add_entity(entity):
                        count_e += 1
                
                # 5. 存入图数据库 - 关系
                for r_data in relations:
                    raw_from = r_data["from"]
                    raw_to = r_data["to"]
                    
                    # 使用映射后的确定性ID，如果找不到（说明LLM生成了幻觉ID），则跳过
                    real_from = id_map.get(raw_from)
                    real_to = id_map.get(raw_to)
                    
                    if real_from and real_to:
                        relation = Relation(
                            from_entity=real_from,
                            to_entity=real_to,
                            relation_type=r_data["type"],
                            properties=r_data.get("properties")
                        )
                        if self.semantic_memory.add_relation(relation):
                            count_r += 1
                    else:
                        pass # 关联的实体未在此次提取中定义，暂不处理复杂情况
                        
                if count_e > 0 or count_r > 0:
                    print(f"[Learning] 成功习得: {count_e} 个新实体, {count_r} 条新关系")
                    
            except json.JSONDecodeError:
                print(f"[Learning] 提取失败: LLM 返回格式错误")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Learning] 学习过程发生错误: {e}")

    ##进行交互的话只执行单步
    def step(self,input_text:str,finish=False,current_step=0,workmemories=None,**kwargs):
        print(f"\n---------第{current_step}步-------\n")

        # 1.构建提示词
        if workmemories is None:
            history_str='None'
        else:
            intent_list = []
            for intent in workmemories:
                 intent_list.extend(intent.get("intents", []))
            history_str = "\n".join(intent_list)
        
        # --- 语义记忆增强 ---
        # 尝试查找输入中的实体并获取关联信息
        # 简单的策略：使用jieba提取名词作为搜索关键词
        keywords = jieba.cut(input_text)
        semantic_context = []
        if self.semantic_memory.graph_store:
            for kw in keywords:
                if len(kw) > 1: # 忽略单字
                    # 在图谱中搜索实体 (实际上现在存进去的是确定性ID，但search_entities_by_name是按名称正则搜索，所以依然有效)
                    entities = self.semantic_memory.graph_store.search_entities_by_name(kw, limit=1)
                    for entity in entities:
                        # 获取该实体的关联信息
                        relations = self.semantic_memory.graph_store.get_entity_relationships(entity['id'])
                        if relations:
                            rel_desc = f"知识图谱信息: {entity.get('name')} -> "
                            rel_strs = []
                            for r in relations[:3]: # 只取前3个关系
                                rel_type = r['relationship'].get('type')
                                target = r['other_entity'].get('name')
                                rel_strs.append(f"{rel_type} {target}")
                            semantic_context.append(rel_desc + ", ".join(rel_strs))
        
        if semantic_context:
            history_str += "\n[Semantic Context]:\n" + "\n".join(semantic_context)
            print(f"语义增强: 检索到 {len(semantic_context)} 条相关图谱知识")
        # ------------------

        prompt = self.prompt_template.format(question=input_text, history=history_str)

        # 2调用大模型
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)

        # 3解析输出
        thought, action = self._parse_output(response)
        return thought, action
        
    def run(self,input_text:str,**kwargs):
        """运行智能体"""
        # self.current_history=[]

        current_step = 0

        print(f"开始处理问题:{input_text}")
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n---------第{current_step}步-------\n")

            #1.构建提示词
            history_str="\n".join(self.current_history)
            prompt = self.prompt_template.format(question=input_text,history=history_str)

            #2调用大模型
            messages = [{"role":"user","content":prompt}]
            response = self.llm.invoke(messages)

            #3解析输出
            thought,action  = self._parse_output(response)

            #检查状态决定下一步
            if action.lower()=="finish":

                self.add_message(Message(content=input_text,role="user"))
                self.add_message(Message(content=thought,role="assistant"))
                # Trigger Learning
                self.learn_from_interaction(input_text, thought)
                return thought
            elif action.lower()=="continue":
                self.add_message(Message(content=input_text,role="user"))
                self.add_message(Message(content=thought,role="assistant"))
                # Trigger Learning
                self.learn_from_interaction(input_text, thought)
                return thought
            else:
                print("未能正确解析到数据，请检查提示词要求")

            print(thought)

        #达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务"
        self.add_message(Message(content=final_answer,role="assistant"))
        self.add_message(Message(content=input_text,role="user"))

        return final_answer







    def _parse_output(self,response):
        try:
            data = json.loads(response)
            action = data.get("type", "continue") # Default to continue if missing
            thought = data.get("message", response) # Default to raw response
            return thought,action
        except json.JSONDecodeError:
            return response, "continue" # Fallback for non-JSON response


    def cli(self):
        print("欢迎使用SmartMenuAgent(输入exit/quit退出)\n")
        self.workmemory=WorkingMemory(MemoryConfig())

        textprocess = TextProcess(llm=self.llm)
        user_input = input("请输入：").strip()
        if user_input.lower() in ["exit","quit"]:
            print("再见")
            return
        current_step=0
        """优化"""
        user_id = self.sessionuserid.user_id
        # content_mem =textprocess.content_to_memory(text=user_input,user_id=user_id)
        # self.workmemory.add(content_mem)
        #=======================
        # self.current_history.append(f"User:{user_input}")
        self.add_message(Message(content="\n".join(user_input), role="user"))
        #self.current_history.append(user_input)
        while current_step<self.max_steps:
            # if self.workmemory.memories:
            #     print(type(self.workmemory.memories))
            thought, action = self.step(input_text=user_input,current_step=current_step,workmemories = self.workmemory.memories)
            self.current_history.append(f"User:{user_input}")
            self.add_message(Message(content=thought, role="assistant"))
            self.current_history.append(f"Assistant:{thought}")
            
            # --- 每次回复后，触发学习流程 ---
            self.learn_from_interaction(user_input, thought)
            # ---------------------------

            if action.lower()=="continue":
                print(thought)
                user_input_implement=input("请补全上述信息").strip()
                user_input = user_input_implement # 更新user_input供下一轮学习使用
                
                self.current_history.append(f"User:{user_input_implement}")
                self.workmemory.add(textprocess.content_to_memory(text=f"User:{user_input},Assistant:{thought},User_Implements:{user_input_implement}",user_id=user_id))
                self.add_message(Message(content=user_input_implement,role="user"))
                current_step+=1
                continue
            else:
                self.workmemory.add(textprocess.content_to_memory(text=f"User:{user_input},Assistant:{thought}",user_id=user_id))
                print(thought)
                break


if __name__ == "__main__":
    #回答推理llm
    llm = LLM(model="gpt-5-mini",provider="chatanywhere")
    agent = SmartMenuAgent(llm)
    #用户意图抽取llm

    agent.cli()
