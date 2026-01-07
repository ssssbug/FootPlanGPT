import datetime
from typing import Literal, Optional, Any, Dict
from pydantic import BaseModel
from datetime import datetime



MessageRole=Literal["user","system","assistant","tool"]
class Message(BaseModel):
    """消息类"""
    content:str
    role:MessageRole
    timestamp:datetime = None
    metadata:Optional[Dict[str,Any]]=None,


    ###转换为OpenAI统一的信息格式
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role":self.role,
            "content":self.content,
        }
    #简化输出格式[role][content]
    def __str__(self)->str:
        return f"[{self.role}] {self.content}]"
