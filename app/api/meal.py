#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
食谱 API - 一周食谱生成与查询

提供静态食谱查询和动态 Agent 生成两种模式
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

# 导入原有模型
from app.schemas.meal import WeeklyMealResponse
from app.services.planner import plan_weekly_meal

router = APIRouter(
    prefix="/meal",
    tags=["meal"]
)


# ========================================
# 扩展模型
# ========================================

class MealGenerateRequest(BaseModel):
    """动态生成食谱请求"""
    preferences: Optional[str] = Field(None, description="用户偏好描述，如 '清淡,低脂,高蛋白'")
    allergies: Optional[List[str]] = Field(None, description="过敏原列表，如 ['海鲜', '花生']")
    city: Optional[str] = Field("北京", description="城市，用于获取天气")
    budget: Optional[str] = Field(None, description="预算等级: low/medium/high")
    health_goal: Optional[str] = Field(None, description="健康目标: 减脂/增肌/养生/普通")


class MealGenerateResponse(BaseModel):
    """动态生成食谱响应"""
    session_id: str = Field(..., description="用于后续追问的会话ID")
    weekly_plan: str = Field(..., description="Agent 生成的一周食谱文本")
    weather_summary: Optional[str] = Field(None, description="天气概况")
    ingredient_tips: Optional[str] = Field(None, description="食材推荐")
    generated_at: str = Field(..., description="生成时间")


class DailyMealDetail(BaseModel):
    """单日详细食谱"""
    date: str
    breakfast: List[str]
    lunch: List[str]
    dinner: List[str]


class WeeklyMealDetail(BaseModel):
    """一周详细食谱"""
    week: Dict[str, DailyMealDetail]
    generated_at: str
    source: str  # "static" or "agent"


# ========================================
# API 端点
# ========================================

@router.get("/weekly-meal", response_model=WeeklyMealResponse)
def get_weekly_meal():
    """
    获取一周三餐推荐 (静态模式)
    
    返回预设的示例食谱，适合快速测试
    """
    return plan_weekly_meal()


@router.post("/generate", response_model=MealGenerateResponse)
async def generate_weekly_meal(request: MealGenerateRequest):
    """
    动态生成一周食谱 (Agent 模式)
    
    根据用户偏好、天气、菜价等动态生成个性化食谱
    """
    try:
        # 导入 Agent 和工具
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from agent.smart_menu_agent import SmartMenuAgent
        from llm.select_llm import LLM
        from memory.WorkingMemory import WorkingMemory
        from memory.baseMemory import MemoryConfig
        
        # 构建用户请求
        user_prompt_parts = ["请为我生成一份科学合理的一周食谱"]
        
        if request.preferences:
            user_prompt_parts.append(f"，我的饮食偏好是: {request.preferences}")
        if request.allergies:
            user_prompt_parts.append(f"，我对以下食材过敏需要避免: {', '.join(request.allergies)}")
        if request.health_goal:
            user_prompt_parts.append(f"，我的健康目标是: {request.health_goal}")
        if request.budget:
            budget_desc = {"low": "经济实惠", "medium": "中等", "high": "不限预算"}.get(request.budget, "中等")
            user_prompt_parts.append(f"，预算: {budget_desc}")
        
        user_message = "".join(user_prompt_parts)
        
        # 创建 Agent
        llm = LLM(model="gpt-4o-mini", provider="chatanywhere")
        agent = SmartMenuAgent(llm=llm)
        agent.user_city = request.city or "北京"
        agent.workmemory = WorkingMemory(MemoryConfig())
        
        # 运行 Agent
        response = agent.run(user_message)
        
        # 获取上下文信息
        weather_summary = None
        ingredient_tips = None
        try:
            from tools import get_weather, get_seasonal_ingredients
            
            weather_data = get_weather(agent.user_city, days=3)
            if weather_data.get("forecasts"):
                first_day = weather_data["forecasts"][0]
                weather_summary = f"{agent.user_city}: {first_day.get('day_weather', '未知')}, {first_day.get('temp_min', '?')}~{first_day.get('temp_max', '?')}°C"
            
            ingredient_data = get_seasonal_ingredients()
            ingredient_tips = ingredient_data.get("seasonal_tips", "")
        except:
            pass
        
        return MealGenerateResponse(
            session_id="",  # 单次生成无需持久化会话
            weekly_plan=response,
            weather_summary=weather_summary,
            ingredient_tips=ingredient_tips,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"食谱生成失败: {str(e)}")


@router.get("/ingredients")
async def get_seasonal_ingredients(
    category: Optional[str] = Query(None, description="类别: vegetable/meat/seafood/all"),
    budget_friendly: bool = Query(False, description="是否只返回性价比高的食材")
):
    """
    获取当季食材推荐
    
    基于当前月份返回时令食材及价格
    """
    try:
        from tools import get_ingredient_prices
        
        result = get_ingredient_prices(
            category=category or "all",
            budget_friendly=budget_friendly
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取食材信息失败: {str(e)}")


@router.get("/weather")
async def get_weather_forecast(
    city: str = Query("北京", description="城市名称"),
    days: int = Query(7, description="预报天数 (1-7)", ge=1, le=7)
):
    """
    获取天气预报
    
    返回指定城市的天气预报及饮食建议
    """
    try:
        from tools import get_weather
        
        result = get_weather(city, days=days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取天气信息失败: {str(e)}")