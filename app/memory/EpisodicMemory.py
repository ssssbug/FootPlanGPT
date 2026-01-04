"""
情景记忆实现
提供:
- 具体交互事件存储
- 时间序列组织
- 上下文丰富的记忆
- 模式识别能力

"""
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from app.memory.baseMemory import BaseMemory, MemoryConfig


class Episode:
    """情景记忆中的单个场景
    """
    def __init__(self,
                 episode_id:str,
                 user_id:str,
                 session_id:str,
                 timestamp:datetime,
                 content:str,
                 context:Dict[str,Any],
                 outcome:Optional[str]=None,
                 importance:float=0.5
                 ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance


class EpisodicMemory(BaseMemory):
    """情景记忆实现
    特点:
    - 存储具体的交互事件
    - 包含丰富的上下文信息
    - 按时间序列组织
    - 支持模式识别和回溯
    """
    def __init__(self,config:MemoryConfig,storage_backend = None):
        super.__init__(config,storage_backend)

        #本地缓存(内存)
        self.episodes:List[Episode] = []
        self.sessions:Dict[str,List[str]]={}#session_id->spisode_ids

        #模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        #权威文档存储(PostgreSQL)
        db_dir = self.config.storage_path if hasattr(self.config,"storage_path") else ".memory_data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir,"memory.db")
        self.doc_store = PostgreSQLDocumentStore(db_path=db_path)

        #统一嵌入模型(多语言，默认384维)
        self.embedder = get_text_embedder()



