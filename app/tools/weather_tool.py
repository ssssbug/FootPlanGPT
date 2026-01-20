#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
天气工具 - 获取未来一周天气预报

支持的天气 API:
1. 和风天气 (QWeather) - 推荐，国内稳定
2. OpenWeatherMap - 国际通用
3. 彩云天气 (Caiyun) - 分钟级预报

使用前需要配置环境变量:
- QWEATHER_API_KEY: 和风天气 API Key
- OPENWEATHERMAP_API_KEY: OpenWeatherMap API Key (可选)
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests

from .base import MCPTool, ToolDefinition, ToolParameter, register_tool


@register_tool
class WeatherTool(MCPTool):
    """获取未来一周天气预报"""
    
    # API 配置
    QWEATHER_BASE_URL = "https://devapi.qweather.com/v7"
    OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    # 城市 ID 映射 (和风天气)
    CITY_IDS = {
        "北京": "101010100",
        "上海": "101020100",
        "广州": "101280101",
        "深圳": "101280601",
        "杭州": "101210101",
        "成都": "101270101",
        "武汉": "101200101",
        "南京": "101190101",
        "西安": "101110101",
        "重庆": "101040100",
    }
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_weather",
            description="获取指定城市未来7天的天气预报，包括温度、天气状况、湿度、风力等信息。可用于根据天气推荐合适的食谱。",
            parameters=[
                ToolParameter(
                    name="city",
                    type="string",
                    description="城市名称，如 '北京', '上海', '广州' 等",
                    required=True
                ),
                ToolParameter(
                    name="days",
                    type="number",
                    description="预报天数 (1-7)，默认7天",
                    required=False,
                    default=7
                )
            ]
        )
    
    def execute(self, city: str, days: int = 7, **kwargs) -> Dict[str, Any]:
        """执行天气查询"""
        days = min(max(1, days), 7)  # 限制1-7天
        
        # 尝试调用真实 API
        api_key = os.getenv("QWEATHER_API_KEY")
        if api_key:
            try:
                return self._fetch_qweather(city, days, api_key)
            except Exception as e:
                print(f"[Weather] API call failed: {e}, using mock data")
        
        # 无 API Key 或调用失败，返回模拟数据
        return self._generate_mock_weather(city, days)
    
    def _fetch_qweather(self, city: str, days: int, api_key: str) -> Dict[str, Any]:
        """调用和风天气 API"""
        # 获取城市 ID
        city_id = self.CITY_IDS.get(city)
        if not city_id:
            # 尝试通过城市搜索 API 获取
            search_url = f"{self.QWEATHER_BASE_URL}/city/lookup"
            resp = requests.get(search_url, params={
                "location": city,
                "key": api_key
            }, timeout=10)
            data = resp.json()
            if data.get("code") == "200" and data.get("location"):
                city_id = data["location"][0]["id"]
            else:
                return {"error": f"城市 '{city}' 未找到", "city": city}
        
        # 获取7天预报
        forecast_url = f"{self.QWEATHER_BASE_URL}/weather/7d"
        resp = requests.get(forecast_url, params={
            "location": city_id,
            "key": api_key
        }, timeout=10)
        data = resp.json()
        
        if data.get("code") != "200":
            return {"error": f"天气查询失败: {data.get('code')}", "city": city}
        
        # 解析天气数据
        forecasts = []
        for day in data.get("daily", [])[:days]:
            forecasts.append({
                "date": day["fxDate"],
                "day_weather": day["textDay"],
                "night_weather": day["textNight"],
                "temp_max": int(day["tempMax"]),
                "temp_min": int(day["tempMin"]),
                "humidity": int(day["humidity"]),
                "wind_dir": day["windDirDay"],
                "wind_scale": day["windScaleDay"],
                "uv_index": day.get("uvIndex", "N/A"),
                "suggestion": self._get_food_suggestion(
                    int(day["tempMax"]), 
                    int(day["tempMin"]),
                    day["textDay"]
                )
            })
        
        return {
            "city": city,
            "update_time": data.get("updateTime", datetime.now().isoformat()),
            "forecasts": forecasts,
            "source": "和风天气"
        }
    
    def _generate_mock_weather(self, city: str, days: int) -> Dict[str, Any]:
        """生成模拟天气数据 (用于演示/测试)"""
        import random
        
        weather_types = ["晴", "多云", "阴", "小雨", "大雨", "雪"]
        
        forecasts = []
        base_date = datetime.now()
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            temp_max = random.randint(15, 35)
            temp_min = temp_max - random.randint(5, 12)
            day_weather = random.choice(weather_types)
            
            forecasts.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_weather": day_weather,
                "night_weather": random.choice(weather_types),
                "temp_max": temp_max,
                "temp_min": temp_min,
                "humidity": random.randint(40, 90),
                "wind_dir": random.choice(["东风", "南风", "西风", "北风", "东北风"]),
                "wind_scale": f"{random.randint(1, 4)}级",
                "uv_index": random.randint(1, 10),
                "suggestion": self._get_food_suggestion(temp_max, temp_min, day_weather)
            })
        
        return {
            "city": city,
            "update_time": datetime.now().isoformat(),
            "forecasts": forecasts,
            "source": "模拟数据 (请配置 QWEATHER_API_KEY 获取真实天气)"
        }
    
    def _get_food_suggestion(self, temp_max: int, temp_min: int, weather: str) -> str:
        """根据天气生成饮食建议"""
        suggestions = []
        
        # 温度建议
        if temp_max >= 30:
            suggestions.append("高温天气，建议清淡饮食，多吃凉拌菜、绿豆汤等解暑食物")
        elif temp_max >= 20:
            suggestions.append("温度适宜，可以均衡搭配各类食材")
        elif temp_max >= 10:
            suggestions.append("天气转凉，建议食用温热食物，如炖汤、火锅")
        else:
            suggestions.append("天气寒冷，推荐高热量滋补食物，如羊肉汤、姜汤")
        
        # 天气建议
        if "雨" in weather:
            suggestions.append("雨天湿气重，建议少油腻，可加入薏米、红豆祛湿")
        elif "雪" in weather:
            suggestions.append("雪天寒冷，推荐热汤和驱寒食材如姜、辣椒")
        elif "晴" in weather:
            suggestions.append("晴天适合户外活动，可准备便携午餐")
        
        return "；".join(suggestions)


# 便捷调用函数
def get_weather(city: str, days: int = 7) -> Dict[str, Any]:
    """获取天气预报的便捷函数"""
    tool = WeatherTool()
    return tool.execute(city=city, days=days)


if __name__ == "__main__":
    # 测试
    result = get_weather("北京", 7)
    print(json.dumps(result, ensure_ascii=False, indent=2))
