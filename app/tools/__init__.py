#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP 工具包初始化

导入此模块会自动注册所有工具
"""

from .base import (
    MCPTool,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    tool_registry,
    register_tool
)

# 导入工具以触发注册
from .weather_tool import WeatherTool, get_weather
from .ingredient_price_tool import IngredientPriceTool, get_ingredient_prices, get_seasonal_ingredients


__all__ = [
    # 基础类
    "MCPTool",
    "ToolDefinition", 
    "ToolParameter",
    "ToolRegistry",
    "tool_registry",
    "register_tool",
    
    # 工具类
    "WeatherTool",
    "IngredientPriceTool",
    
    # 便捷函数
    "get_weather",
    "get_ingredient_prices",
    "get_seasonal_ingredients",
]
