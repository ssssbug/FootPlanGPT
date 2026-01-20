#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聊天 API - 对话式食谱生成

提供与 SmartMenuAgent 交互的 RESTful API 接口
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

# 导入 Agent
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.smart_menu_agent import SmartMenuAgent
from llm.select_llm import LLM
from memory.WorkingMemory import WorkingMemory
from memory.baseMemory import MemoryConfig

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


# ========================================
# 请求/响应模型
# ========================================

class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., description="用户消息", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话ID，留空则创建新会话")
    city: Optional[str] = Field("北京", description="用户所在城市，用于获取天气")


class ChatResponse(BaseModel):
    """对话响应"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="Agent 回复")
    action_type: str = Field(..., description="动作类型: continue/final")
    timestamp: str = Field(..., description="响应时间")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息 (天气/菜价)")


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    created_at: str
    message_count: int
    last_activity: str


class ContextResponse(BaseModel):
    """环境上下文响应"""
    weather: Dict[str, Any] = Field(..., description="天气信息")
    ingredients: Dict[str, Any] = Field(..., description="食材价格")


# ========================================
# 会话管理 (简单内存存储，生产环境应使用 Redis)
# ========================================

class SessionManager:
    """会话管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions: Dict[str, Dict[str, Any]] = {}
            cls._instance._agents: Dict[str, SmartMenuAgent] = {}
            cls._instance._llm = None
        return cls._instance
    
    def _get_llm(self) -> LLM:
        """获取共享 LLM 实例"""
        if self._llm is None:
            self._llm = LLM(model="gpt-4o-mini", provider="chatanywhere")
        return self._llm
    
    def create_session(self, city: str = "北京") -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # 创建 Agent 实例
        agent = SmartMenuAgent(llm=self._get_llm())
        agent.user_city = city
        agent.workmemory = WorkingMemory(MemoryConfig())
        
        self._sessions[session_id] = {
            "created_at": now,
            "last_activity": now,
            "message_count": 0,
            "city": city,
            "history": []
        }
        self._agents[session_id] = agent
        
        return session_id
    
    def get_agent(self, session_id: str) -> Optional[SmartMenuAgent]:
        """获取会话对应的 Agent"""
        return self._agents.get(session_id)
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, user_msg: str, agent_msg: str):
        """更新会话"""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session["last_activity"] = datetime.now().isoformat()
            session["message_count"] += 1
            session["history"].append({
                "user": user_msg,
                "agent": agent_msg,
                "timestamp": datetime.now().isoformat()
            })
    
    def list_sessions(self) -> List[SessionInfo]:
        """列出所有会话"""
        return [
            SessionInfo(
                session_id=sid,
                created_at=info["created_at"],
                message_count=info["message_count"],
                last_activity=info["last_activity"]
            )
            for sid, info in self._sessions.items()
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            if session_id in self._agents:
                del self._agents[session_id]
            return True
        return False


# 全局会话管理器
session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """依赖注入: 获取会话管理器"""
    return session_manager


# ========================================
# API 端点
# ========================================

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    manager: SessionManager = Depends(get_session_manager)
):
    """
    与 Agent 对话
    
    - 发送消息，Agent 会根据天气、菜价、知识库生成个性化回复
    - 首次请求无需 session_id，系统会自动创建会话
    - 后续请求携带 session_id 以保持上下文
    """
    # 获取或创建会话
    if request.session_id:
        agent = manager.get_agent(request.session_id)
        if not agent:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
        session_id = request.session_id
    else:
        session_id = manager.create_session(city=request.city or "北京")
        agent = manager.get_agent(session_id)
    
    # 更新城市设置
    if request.city:
        agent.user_city = request.city
    
    try:
        # 调用 Agent 进行对话
        session_info = manager.get_session(session_id)
        current_step = session_info.get("message_count", 0)
        
        thought, action = agent.step(
            input_text=request.message,
            current_step=current_step,
            workmemories=agent.workmemory.memories if hasattr(agent, 'workmemory') else None
        )
        
        # 触发学习
        agent.learn_from_interaction(request.message, thought)
        
        # 更新会话
        manager.update_session(session_id, request.message, thought)
        
        # 构建上下文信息 (可选返回)
        context = None
        try:
            from tools import get_weather, get_seasonal_ingredients
            context = {
                "weather_city": agent.user_city,
                "season": get_seasonal_ingredients().get("season", "未知")
            }
        except:
            pass
        
        return ChatResponse(
            session_id=session_id,
            message=thought,
            action_type=action.lower() if action else "continue",
            timestamp=datetime.now().isoformat(),
            context=context
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")


@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions(
    manager: SessionManager = Depends(get_session_manager)
):
    """列出所有活跃会话"""
    return manager.list_sessions()


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager)
):
    """删除指定会话"""
    if manager.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="会话不存在")


@router.get("/context", response_model=ContextResponse)
async def get_context(
    city: str = "北京"
):
    """
    获取当前环境上下文 (天气 + 食材价格)
    
    可用于前端展示实时数据
    """
    try:
        from tools import get_weather, get_seasonal_ingredients
        
        weather = get_weather(city, days=7)
        ingredients = get_seasonal_ingredients()
        
        return ContextResponse(
            weather=weather,
            ingredients=ingredients
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取上下文失败: {str(e)}")


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    }
