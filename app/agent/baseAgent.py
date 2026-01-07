from abc import ABC,abstractmethod
from typing import Optional, List

from app.agent.agent import Agent
from app.message.message import Message


class BaseAgent(ABC):
    def __init__(self,
                 name:str,
                 llm:Agent,
                 system_prompt:Optional[str]=None,
                 Config:Optional[str]=None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.Config = Config
        self._history:list[Message]=[]

    @abstractmethod
    def run(self,input_text:str,**kwargs) -> str:
        """运行agent"""
        pass
    def add_message(self,message:Message):
        """添加历史信息"""
        self._history.append(message)
        # pass
    def clear_history(self):
        """清空历史记录"""
        # self._history.clear()
        pass

    def get_history(self)->List[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self)->str:
        return f"Agent (name={self.name},provider={self.llm.provider})"

