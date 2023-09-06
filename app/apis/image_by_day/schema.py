from pydantic import BaseModel, Field
from typing import Optional


class FeatureModelImage(BaseModel):
    day_month_year: str = Field(..., description="The time want to search for")
    time_period: Optional[str] = Field(..., description="time period, morning or the afternoon")
    hour: Optional[str] = Field(..., description="hour")

    class Config:
        orm_mode = True

    def __str__(self):
        return f"{self.day_month_year}," \
               f"{self.time_period}, {self.hour},"
