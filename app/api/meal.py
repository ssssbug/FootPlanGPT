from fastapi import APIRouter
from app.schemas.meal import WeeklyMealResponse
from app.services.planner import plan_weekly_meal
router = APIRouter(
    prefix="/meal",
    tags =["meal"]
)

@router.get("/weekly-meal",response_model=WeeklyMealResponse)
def get_weekly_meal():

    """获取一周三餐推荐"""
    return plan_weekly_meal()