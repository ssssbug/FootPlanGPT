#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
食材价格工具 - 获取当季食材价格

数据来源:
1. 批发市场价格 (模拟/接口)
2. 季节性调整
3. 地区差异

使用前可配置:
- VEGETABLE_PRICE_API_KEY: 如有真实接口可配置
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import random

from .base import MCPTool, ToolDefinition, ToolParameter, register_tool


# 基础食材价格数据 (元/斤) - 基准价格
BASE_PRICES = {
    # 蔬菜类
    "白菜": 1.5,
    "菠菜": 4.0,
    "生菜": 3.5,
    "油麦菜": 4.0,
    "小白菜": 2.5,
    "大白菜": 1.2,
    "青菜": 2.0,
    "空心菜": 3.0,
    "韭菜": 4.5,
    "芹菜": 3.0,
    "西兰花": 6.0,
    "花菜": 4.0,
    "卷心菜": 2.0,
    "茄子": 3.5,
    "黄瓜": 3.0,
    "西红柿": 4.0,
    "番茄": 4.0,
    "土豆": 2.5,
    "红薯": 3.0,
    "山药": 8.0,
    "胡萝卜": 2.5,
    "白萝卜": 1.5,
    "洋葱": 2.0,
    "大葱": 3.5,
    "生姜": 12.0,
    "大蒜": 8.0,
    "蒜苗": 5.0,
    "青椒": 4.0,
    "辣椒": 5.0,
    "豆角": 5.0,
    "四季豆": 5.5,
    "豌豆": 8.0,
    "毛豆": 6.0,
    "玉米": 3.0,
    "南瓜": 2.0,
    "冬瓜": 1.5,
    "丝瓜": 4.0,
    "苦瓜": 4.5,
    "莲藕": 6.0,
    "竹笋": 10.0,
    "香菇": 15.0,
    "木耳": 25.0,
    "金针菇": 6.0,
    "平菇": 5.0,
    "豆腐": 3.0,
    "豆皮": 6.0,
    
    # 肉类
    "猪肉": 15.0,
    "五花肉": 18.0,
    "排骨": 28.0,
    "猪蹄": 22.0,
    "牛肉": 45.0,
    "牛腩": 42.0,
    "羊肉": 48.0,
    "羊排": 55.0,
    "鸡肉": 12.0,
    "鸡胸肉": 14.0,
    "鸡腿": 13.0,
    "鸡翅": 18.0,
    "鸭肉": 16.0,
    
    # 海鲜类
    "带鱼": 18.0,
    "鲫鱼": 12.0,
    "草鱼": 10.0,
    "鲈鱼": 25.0,
    "三文鱼": 60.0,
    "虾": 35.0,
    "基围虾": 40.0,
    "螃蟹": 45.0,
    "鱿鱼": 25.0,
    "贝类": 20.0,
    
    # 蛋奶类
    "鸡蛋": 5.5,
    "鸭蛋": 7.0,
    "牛奶": 6.0,  # 元/升
    
    # 主食类
    "大米": 3.0,
    "面粉": 2.5,
    "糯米": 5.0,
    "小米": 6.0,
    "燕麦": 8.0,
}

# 季节性调整因子 (月份 -> 食材类别 -> 调整系数)
SEASONAL_FACTORS = {
    # 春季 (3-5月)
    3: {"菠菜": 0.8, "韭菜": 0.7, "竹笋": 0.6, "草莓": 0.7},
    4: {"菠菜": 0.7, "蒜苗": 0.8, "豌豆": 0.8},
    5: {"黄瓜": 0.8, "番茄": 0.8, "茄子": 0.9},
    
    # 夏季 (6-8月)
    6: {"黄瓜": 0.6, "西红柿": 0.6, "茄子": 0.7, "豆角": 0.7, "丝瓜": 0.6, "苦瓜": 0.6},
    7: {"黄瓜": 0.5, "西红柿": 0.5, "茄子": 0.6, "玉米": 0.7, "毛豆": 0.7},
    8: {"黄瓜": 0.6, "西红柿": 0.6, "冬瓜": 0.5, "南瓜": 0.6},
    
    # 秋季 (9-11月)
    9: {"土豆": 0.7, "红薯": 0.7, "南瓜": 0.5, "莲藕": 0.7},
    10: {"白菜": 0.6, "萝卜": 0.6, "山药": 0.8, "香菇": 0.8},
    11: {"大白菜": 0.5, "白萝卜": 0.5, "羊肉": 0.9},
    
    # 冬季 (12-2月)
    12: {"大白菜": 0.4, "白萝卜": 0.4, "羊肉": 0.85, "牛肉": 0.95},
    1: {"大白菜": 0.5, "萝卜": 0.5, "羊肉": 0.85},
    2: {"菠菜": 1.3, "韭菜": 1.2, "竹笋": 1.5},  # 早春涨价
}


@register_tool
class IngredientPriceTool(MCPTool):
    """获取食材价格信息"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_ingredient_prices",
            description="获取当前食材价格，支持查询特定食材或获取当季推荐食材列表。价格会根据季节动态调整。",
            parameters=[
                ToolParameter(
                    name="ingredients",
                    type="array",
                    description="要查询的食材列表，如 ['白菜', '猪肉', '鸡蛋']。为空则返回当季推荐食材。",
                    required=False,
                    default=[]
                ),
                ToolParameter(
                    name="category",
                    type="string",
                    description="食材类别筛选: 'vegetable'(蔬菜), 'meat'(肉类), 'seafood'(海鲜), 'all'(全部)",
                    required=False,
                    default="all",
                    enum=["vegetable", "meat", "seafood", "egg", "staple", "all"]
                ),
                ToolParameter(
                    name="budget_friendly",
                    type="boolean",
                    description="是否只推荐性价比高的食材 (价格低于均值)",
                    required=False,
                    default=False
                )
            ]
        )
    
    # 食材分类
    CATEGORIES = {
        "vegetable": ["白菜", "菠菜", "生菜", "油麦菜", "小白菜", "大白菜", "青菜", "空心菜", 
                      "韭菜", "芹菜", "西兰花", "花菜", "卷心菜", "茄子", "黄瓜", "西红柿",
                      "番茄", "土豆", "红薯", "山药", "胡萝卜", "白萝卜", "洋葱", "大葱",
                      "生姜", "大蒜", "蒜苗", "青椒", "辣椒", "豆角", "四季豆", "豌豆",
                      "毛豆", "玉米", "南瓜", "冬瓜", "丝瓜", "苦瓜", "莲藕", "竹笋",
                      "香菇", "木耳", "金针菇", "平菇", "豆腐", "豆皮"],
        "meat": ["猪肉", "五花肉", "排骨", "猪蹄", "牛肉", "牛腩", "羊肉", "羊排",
                 "鸡肉", "鸡胸肉", "鸡腿", "鸡翅", "鸭肉"],
        "seafood": ["带鱼", "鲫鱼", "草鱼", "鲈鱼", "三文鱼", "虾", "基围虾", "螃蟹", "鱿鱼", "贝类"],
        "egg": ["鸡蛋", "鸭蛋", "牛奶"],
        "staple": ["大米", "面粉", "糯米", "小米", "燕麦"]
    }
    
    def execute(
        self, 
        ingredients: List[str] = None, 
        category: str = "all",
        budget_friendly: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """执行价格查询"""
        current_month = datetime.now().month
        
        # 确定要查询的食材
        if ingredients:
            query_ingredients = ingredients
        elif category != "all":
            query_ingredients = self.CATEGORIES.get(category, [])
        else:
            # 返回当季推荐 (价格有折扣的)
            query_ingredients = self._get_seasonal_recommendations(current_month)
        
        # 计算价格
        results = []
        for name in query_ingredients:
            base_price = BASE_PRICES.get(name)
            if base_price is None:
                continue
            
            # 应用季节性调整
            factor = SEASONAL_FACTORS.get(current_month, {}).get(name, 1.0)
            # 添加随机波动 (±5%)
            fluctuation = 1 + (random.random() - 0.5) * 0.1
            current_price = round(base_price * factor * fluctuation, 2)
            
            # 判断是否当季
            is_seasonal = factor < 1.0
            
            results.append({
                "name": name,
                "price": current_price,
                "unit": "元/斤",
                "is_seasonal": is_seasonal,
                "price_trend": self._get_price_trend(factor),
                "category": self._get_category(name)
            })
        
        # 按价格排序
        results.sort(key=lambda x: x["price"])
        
        # 性价比筛选
        if budget_friendly:
            avg_price = sum(r["price"] for r in results) / len(results) if results else 0
            results = [r for r in results if r["price"] <= avg_price]
        
        return {
            "query_time": datetime.now().isoformat(),
            "month": current_month,
            "season": self._get_season(current_month),
            "total_count": len(results),
            "prices": results,
            "seasonal_tips": self._get_seasonal_tips(current_month),
            "source": "模拟数据 (基于季节性调整)"
        }
    
    def _get_seasonal_recommendations(self, month: int) -> List[str]:
        """获取当季推荐食材"""
        seasonal = SEASONAL_FACTORS.get(month, {})
        # 返回当季食材 (factor < 1.0)
        recommendations = [name for name, factor in seasonal.items() if factor < 1.0]
        
        # 如果没有特别推荐，返回基础蔬菜
        if not recommendations:
            recommendations = ["白菜", "土豆", "胡萝卜", "鸡蛋", "猪肉", "豆腐"]
        
        return recommendations
    
    def _get_price_trend(self, factor: float) -> str:
        """获取价格趋势描述"""
        if factor < 0.7:
            return "大幅下降 ↓↓"
        elif factor < 0.9:
            return "小幅下降 ↓"
        elif factor > 1.2:
            return "大幅上涨 ↑↑"
        elif factor > 1.05:
            return "小幅上涨 ↑"
        else:
            return "价格稳定 →"
    
    def _get_category(self, name: str) -> str:
        """获取食材类别"""
        for cat, items in self.CATEGORIES.items():
            if name in items:
                return cat
        return "other"
    
    def _get_season(self, month: int) -> str:
        """获取季节"""
        if month in [3, 4, 5]:
            return "春季"
        elif month in [6, 7, 8]:
            return "夏季"
        elif month in [9, 10, 11]:
            return "秋季"
        else:
            return "冬季"
    
    def _get_seasonal_tips(self, month: int) -> str:
        """获取季节性饮食建议"""
        tips = {
            "春季": "春季万物复苏，推荐多吃时令蔬菜如菠菜、韭菜、春笋，有助于养肝护肝",
            "夏季": "夏季炎热，建议多吃清热解暑食物如黄瓜、西红柿、冬瓜、绿豆，少油腻",
            "秋季": "秋季干燥，推荐润燥食物如莲藕、银耳、百合、梨，适当进补",
            "冬季": "冬季寒冷，推荐温补食物如羊肉、牛肉、白萝卜炖汤，增强抵抗力"
        }
        return tips.get(self._get_season(month), "")


# 便捷调用函数
def get_ingredient_prices(
    ingredients: List[str] = None, 
    category: str = "all",
    budget_friendly: bool = False
) -> Dict[str, Any]:
    """获取食材价格的便捷函数"""
    tool = IngredientPriceTool()
    return tool.execute(
        ingredients=ingredients,
        category=category,
        budget_friendly=budget_friendly
    )


def get_seasonal_ingredients() -> Dict[str, Any]:
    """获取当季食材"""
    return get_ingredient_prices(category="all", budget_friendly=True)


if __name__ == "__main__":
    # 测试
    print("=" * 50)
    print("测试1: 查询特定食材")
    result = get_ingredient_prices(["白菜", "猪肉", "鸡蛋", "虾"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 50)
    print("测试2: 获取当季推荐")
    result = get_seasonal_ingredients()
    print(json.dumps(result, ensure_ascii=False, indent=2))
