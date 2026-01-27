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
from memory.EpisodicMemory import EpisodicMemory # Added EpisodicMemory
from memory.baseMemory import MemoryItem, MemoryConfig
from memory.Semantic import SemanticMemory, Entity, Relation # Import Entity/Relation
from message.message import Message
from prompt.default_prompt import DEFAULT_REACT_TEMPLATE, INTENT_PROMPT, MEMORY_EXTRACTION_PROMPT, REFLECTION_PROMPT # Added Prompt
from schemas.AgentState import AgentState
from utils.text_process import SessionUserId
from utils.text_process import TextProcess
#环境加载，加载llm
load_dotenv()

# RAG Pipeline import
try:
    from services.rag_pipeline import create_rag_pipeline
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("[WARNING] RAG Pipeline not available")

# MCP Tools import
try:
    from tools import get_weather, get_ingredient_prices, get_seasonal_ingredients
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("[WARNING] MCP Tools not available")

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
        
        # 初始化情景记忆
        self.episodic_memory = EpisodicMemory(MemoryConfig())
        
        # 记录交互轮数用于触发反思
        self.turn_count = 0
        
        # 初始化 RAG Pipeline
        self.rag_pipeline = None
        if RAG_AVAILABLE:
            try:
                self.rag_pipeline = create_rag_pipeline(
                    collection_name="food_recipe_kb",
                    rag_namespace="food_kb",
                    neo4j_enabled=True
                )
                print("[Agent] RAG Pipeline initialized successfully")
            except Exception as e:
                print(f"[Agent] RAG Pipeline initialization failed: {e}")
                self.rag_pipeline = None
        
        # 初始化 MCP 工具数据
        self.user_city = "北京"  # 默认城市，可通过用户偏好更新
        self._weather_cache = None
        self._ingredient_cache = None
        self._cache_time = None
        
        if MCP_AVAILABLE:
            print("[Agent] MCP Tools available")

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

    def reflect_and_consolidate(self, user_id: str):
        """
        [自我反思任务] 
        1. 检索最近的情景记忆
        2. 由 LLM 总结出稳定的用户偏好和知识
        3. 将总结结果存入语义记忆（图谱）
        """
        print(f"\n[Reflection] 正在对用户 {user_id} 的近期表现进行深度反思...")
        
        # 1. 获取最近 10 条交互记录
        recent_episodes = self.episodic_memory.get_timeline(user_id=user_id, Limit=10)
        if len(recent_episodes) < 3:
            print("[Reflection] 记忆不足，暂不执行反思")
            return

        episode_texts = [f"- {e['timestamp']}: {e['content']}" for e in recent_episodes]
        context_str = "\n".join(episode_texts)

        # 2. 调用 LLM 进行反思
        prompt = REFLECTION_PROMPT.format(context=context_str)
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            
            # 清洗并通过与 learn_from_interaction 类似的逻辑入库
            # 这里简化处理，直接复用提取逻辑的解析部分
            self.learn_from_interaction("Reflection Context", response)
            print("[Reflection] 反思完成，已将洞察固化到语义图谱")
        except Exception as e:
            print(f"[Reflection] 反思失败: {e}")

    def _retrieve_rag_context(self, query: str, top_k: int = 3) -> str:
        """
        从 RAG 知识库检索相关上下文
        """
        if not self.rag_pipeline:
            return "(知识库未初始化)"
        
        try:
            results = self.rag_pipeline["search"](query, top_k=top_k)
            if not results:
                return "(未找到相关知识)"
            
            # 格式化检索结果
            context_parts = []
            for i, r in enumerate(results, 1):
                content = r.get("content", r.get("text", ""))
                source = r.get("metadata", {}).get("source_path", "未知来源")
                # 截取前500字符避免过长
                if len(content) > 500:
                    content = content[:500] + "..."
                context_parts.append(f"[{i}] {content}\n   来源: {source}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"[RAG] Search error: {e}")
            return "(检索失败)"

    def _get_weather_context(self) -> str:
        """
        获取天气预报上下文
        """
        if not MCP_AVAILABLE:
            return "(天气服务未启用)"
        
        try:
            # 使用缓存避免频繁调用
            from datetime import datetime, timedelta
            now = datetime.now()
            
            if self._weather_cache and self._cache_time:
                # 缓存 1 小时有效
                if now - self._cache_time < timedelta(hours=1):
                    return self._weather_cache
            
            # 调用天气工具
            weather_data = get_weather(self.user_city, days=7)
            
            if "error" in weather_data:
                return f"(天气查询失败: {weather_data['error']})"
            
            # 格式化天气信息
            lines = [f"城市: {weather_data.get('city', self.user_city)}"]
            lines.append(f"数据来源: {weather_data.get('source', '未知')}")
            lines.append("")
            
            for forecast in weather_data.get("forecasts", []):
                date = forecast.get("date", "")
                day_weather = forecast.get("day_weather", "")
                temp_max = forecast.get("temp_max", "?")
                temp_min = forecast.get("temp_min", "?")
                suggestion = forecast.get("suggestion", "")
                
                lines.append(f"- {date}: {day_weather}, {temp_min}~{temp_max}°C")
                if suggestion:
                    lines.append(f"  建议: {suggestion}")
            
            result = "\n".join(lines)
            
            # 更新缓存
            self._weather_cache = result
            self._cache_time = now
            
            return result
            
        except Exception as e:
            print(f"[MCP] Weather error: {e}")
            return "(天气服务异常)"

    def _get_ingredient_context(self) -> str:
        """
        获取当季食材价格上下文
        """
        if not MCP_AVAILABLE:
            return "(食材价格服务未启用)"
        
        try:
            # 获取当季推荐食材
            price_data = get_seasonal_ingredients()
            
            if "error" in price_data:
                return f"(价格查询失败: {price_data['error']})"
            
            # 格式化价格信息
            lines = [
                f"当前季节: {price_data.get('season', '未知')}",
                f"数据来源: {price_data.get('source', '未知')}",
                "",
                price_data.get("seasonal_tips", ""),
                "",
                "当季推荐食材 (性价比高):"
            ]
            
            # 按类别分组显示
            categories = {}
            for item in price_data.get("prices", []):
                cat = item.get("category", "other")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)
            
            cat_names = {
                "vegetable": "蔬菜",
                "meat": "肉类", 
                "seafood": "海鲜",
                "egg": "蛋奶",
                "staple": "主食",
                "other": "其他"
            }
            
            for cat, items in categories.items():
                if items:
                    cat_name = cat_names.get(cat, cat)
                    item_strs = [f"{i['name']}({i['price']}元/斤)" for i in items[:5]]
                    lines.append(f"  {cat_name}: {', '.join(item_strs)}")
            
            return "\n".join(lines) 
            
        except Exception as e:
            print(f"[MCP] Ingredient price error: {e}")
            return "(食材价格服务异常)"

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
        
        # --- 情景记忆检索 ---
        episodic_context = []
        try:
            # 搜索与当前问题相关的历史瞬间
            episodes = self.episodic_memory.retrieve(input_text, user_id=self.sessionuserid.user_id, limit=3)
            if episodes:
                episodic_context.append("你回忆起之前的零散对话:")
                for e in episodes:
                    episodic_context.append(f"- 在 {e.timestamp.strftime('%Y-%m-%d %H:%M')}: {e.content}")
        except Exception as e:
            print(f"[Episodic] Retrieve error: {e}")

        if episodic_context:
            history_str += "\n[Episodic Context]:\n" + "\n".join(episodic_context)
            print(f"情景增强: 检索到 {len(episodes)} 个相关历史瞬间")
        # ------------------

        # 3. 实时 RAG 检索
        rag_context = self._retrieve_rag_context(input_text)

        prompt = self.prompt_template.format(
            question=input_text, 
            history=history_str,
            weather_context=self._get_weather_context(),
            ingredient_context=self._get_ingredient_context(),
            rag_context=rag_context
        )

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
            rag_context = self._retrieve_rag_context(input_text)
            weather_context = self._get_weather_context()
            ingredient_context = self._get_ingredient_context()
            prompt = self.prompt_template.format(
                question=input_text,
                history=history_str,
                weather_context=weather_context,
                ingredient_context=ingredient_context,
                rag_context=rag_context
            )

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
                
                # 记录情景记忆
                self.episodic_memory.add(MemoryItem(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    content=f"User: {user_input} -> Assistant: {thought} -> User Complement: {user_input_implement}",
                    importance=0.5,
                    metadata={"session_id": "cli_session", "type": "interaction"}
                ))
                
                user_input = user_input_implement # 更新user_input供下一轮学习使用
                
                self.current_history.append(f"User:{user_input_implement}")
                self.workmemory.add(textprocess.content_to_memory(text=f"User:{user_input},Assistant:{thought},User_Implements:{user_input_implement}",user_id=user_id))
                self.add_message(Message(content=user_input_implement,role="user"))
                current_step+=1
                
                # 检查是否触发反思 (每3轮触发一次，演示用)
                self.turn_count += 1
                if self.turn_count % 3 == 0:
                    self.reflect_and_consolidate(user_id)
                
                continue
            else:
                # 记录情景记忆 (最终回复)
                self.episodic_memory.add(MemoryItem(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    content=f"User: {user_input} -> Assistant: {thought}",
                    importance=0.6,
                    metadata={"session_id": "cli_session", "type": "final_interaction"}
                ))
                
                self.workmemory.add(textprocess.content_to_memory(text=f"User:{user_input},Assistant:{thought}",user_id=user_id))
                print(thought)
                
                # 结束时也执行一次反思
                self.reflect_and_consolidate(user_id)
                break


if __name__ == "__main__":
    #回答推理llm
    llm = LLM(model="gpt-5-mini",provider="chatanywhere")
    agent = SmartMenuAgent(llm)
    #用户意图抽取llm

    agent.cli()
