import json
import os
import uuid
from datetime import datetime
from typing import Tuple, Optional, List

import jieba
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from app.agent.agent import Agent
from app.agent.baseAgent import BaseAgent
from app.llm.select_llm import LLM
from app.memory.WorkingMemory import WorkingMemory
from app.memory.baseMemory import MemoryItem, MemoryConfig
from app.message.message import Message
from app.prompt.default_prompt import DEFAULT_REACT_TEMPLATE, INTENT_PROMPT
from app.schemas.AgentState import AgentState
from app.utils.text_process import SessionUserId
from app.utils.text_process import TextProcess
#环境加载，加载llm
load_dotenv()





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
    ##进行交互的话只执行单步
    def step(self,input_text:str,finish=False,current_step=0,workmemories=None,**kwargs):
        print(f"\n---------第{current_step}步-------\n")

        # 1.构建提示词
        if workmemories is None:
            history_str='None'
        else:
            for intent  in workmemories:
                history_str = "\n".join(intent.get("intents"))
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
                return thought
            elif action.lower()=="continue":
                self.add_message(Message(content=input_text,role="user"))
                self.add_message(Message(content=thought,role="assistant"))
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
        data = json.loads(response)
        action = data["type"]
        thought = data["message"]
        return thought,action


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
            if self.workmemory.memories:
                print(type(self.workmemory.memories))
            thought, action = self.step(input_text=user_input,current_step=current_step,workmemories = self.workmemory.memories)
            self.current_history.append(f"User:{user_input}")
            self.add_message(Message(content=thought, role="assistant"))
            self.current_history.append(f"Assistant:{thought}")

            if action.lower()=="continue":
                print(thought)
                user_input_implement=input("请补全上述信息").strip()
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
