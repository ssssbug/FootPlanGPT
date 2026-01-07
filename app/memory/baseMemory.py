"""
记忆系统基础类和配置
"""
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List

from pydantic import BaseModel


class MemoryItem(BaseModel):
    """记忆像数据结构
    重要性:内在价值，是与历史记忆的相关性衡量，加入后不会改变
    priority:在重要性的基础上加入时间衰减和使用频率
    """
    id:str
    content:str
    keyword:List[str]
    memory_type:str
    user_id:str
    timestamp:datetime
    importance:float=0.5
    metadata:Dict[str,Any]={}

    class Config:
        arbitrary_types_allowed = True

class MemoryConfig(BaseModel):
    """记忆系统配置"""
    #存储路径
    storage_path:str='./memory_data'

    #统计显示用的基础配置
    max_capacity:int=100
    importance_threshold:float=0.1
    decay_factor:float=0.95
    max_tokens:int=2000

    #工作记忆特定配置(短期记忆，纯内存存储)
    working_memory_capacity:int = 50
    working_memory_tokens:int = 2000
    working_memory_ttl_minutes:int = 120#2小时
    working_memory_base_impotance:float=0.5

class BaseMemory(ABC):
    """记忆基类
    定义所有记忆类型的通用接口和行为"""

    def __init__(self,config:MemoryConfig,storage_backend=None):
        self.storage_backend=storage_backend
        self.config=config
        self.memory_type = self.__class__.__name__.lower().replace("memory","")


    @abstractmethod
    def add(self,memory_item:MemoryItem):
        """添加记忆"""
        pass
    @abstractmethod
    def retrieve(self,query:str,limit:int=5,**kwargs)->List[MemoryItem]:

        """检索相关记忆
        query:查询内容
        limit:限制返回的数量
        kwargs:其余查询参数
        return:相关记忆列表
        """
        pass
    @abstractmethod
    def update(self,memory_id:str,content:str=None,importance:float=None,metadata:Dict[str,Any]=None)->bool:
        """
        更新记忆
        :param memory_id:记忆id
        :param content: 更新内容
        :param importance: 更新重要性
        :param metadata: 更新元数据
        :return: 是否更新成功
        """
        pass
    @abstractmethod
    def delete(self,memory_id:str)->bool:
        """
        删除记忆
        :param memory_id:记忆id
        :return:是否删除成功
        """
        pass
    @abstractmethod
    def has_memory(self,memory_id:str)->bool:
        """
        根据记忆id检查记忆是否存在
        :param memory_id:memory_id
        :return:
        """
        pass
    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass
    @abstractmethod
    def get_stats(self)->Dict[str, Any]:
        """
        获取记忆统计信息
        :return: 统计信息字典
        """


    def generate_id(self)->str:
        """生成记忆id"""
        return (str(uuid.uuid4()))

    def calculate_importance(self,history_memory:List[MemoryItem],current_memory:MemoryItem)->float:
        """
        计算记忆重要性
        :param content:文本内容
        :param base_impotance:基础重要性
        :return:
        """
        importance = self.config.working_memory_base_impotance
        history_keywords = set()
        current_keywords = set(current_memory.keyword)
        #使用content关键词与当前短期记忆所有关键词重叠度计算jaccard相似度
        if history_memory:
            for memory_item in history_memory:
                keyword = memory_item.keyword
                history_keywords.update(keyword)
            overlap= len(history_keywords & current_keywords)
            union_len = len(history_keywords | current_keywords)

            #计算jaccard相似度
            if overlap>=0 and union_len>0:
                jaccard_sim = overlap/union_len if union_len!=0 else 0
            else:
                jaccard_sim = importance
        else:
            jaccard_sim = importance#第一条记忆重要性为1.0

        return jaccard_sim+importance










































