from pydantic import BaseModel, Field
from typing import Optional, List


class FeatureModelRelevanceSearch(BaseModel):
    image_id: List[str] = Field(..., description="the array of relevant image id")
    query: str = Field(..., description="the query")

    class Config:
        orm_mode = True

    def __str__(self):
        return f"{self.query}," \
               f"{self.image_id}"
