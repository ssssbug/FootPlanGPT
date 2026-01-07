from pydantic import BaseModel, Field
from typing import Dict,List

class RecipeStep(BaseModel):
    step_number: int=Field(...,description="步骤序号")
    instruction: str=Field(...,description="具体操作步骤")



class DailyMeal(BaseModel):
    meal_type:str=Field(...,description="餐别,如:午餐,晚餐")
    dish_name:str=Field(...,description="菜品名称")
    ingredients:List[str]=Field(...,description="所需食材")
    steps:List[RecipeStep]=Field(...,description="烹饪步骤")

class WeeklyMenu(BaseModel):
    menu:List[DailyMeal]=Field(...,description="一周的完整食谱")

