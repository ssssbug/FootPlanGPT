import datetime
from typing import List, Optional, Tuple
from datetime import date
from pydantic import BaseModel

class DailyWeather(BaseModel):
    date:date
    temp_max:Optional[float]
    temp_min:Optional[float]
    weather:str #晴、雨、雪

class WeeklyWeather(BaseModel):
    days:Tuple[date,DailyWeather]




