from pydantic import BaseModel, Field
from typing import Optional, List


class FeatureModelSingleSearch(BaseModel):
    query: str = Field(..., description="concept query that user input")
    semantic_name: Optional[str] = Field(default='', description='semantic_name')

    # semantic_name: Optional[str] = Field(..., description='location')
    # start_hour: Optional[int] = Field(default=0, description='start hour')
    # end_hour: Optional[int] = Field(default=24, description='end hour')
    # is_weekend: Optional[int] = Field(default=None, description='is on weekend or not')

    class Config:
        orm_mode = True

    def __str__(self):
        return f"{self.query}, {self.semantic_name}"


class FeatureModelTemporalSearch(BaseModel):
    query: str = Field(..., description="concept query that user input")
    previous_event: Optional[str] = Field(..., description="query of previous event")
    next_event: Optional[str] = Field(..., description='query of next event')
    time_gap: Optional[int] = Field(default=1, description='time between events')
    semantic_name: Optional[str] = Field(default='', description='semantic name')

    class Config:
        orm_mode = True

    def __str__(self):
        return f"{self.query}," \
               f"{self.previous_event}, {self.next_event}, " \
               f"{self.time_gap}, {self.semantic_name}"


class FeatureModelConversationalSearch(BaseModel):
    query: str = Field(..., description="concept query that user input")
    previous_chat: list = Field(..., description="list of previous chat")

    class Config:
        orm_mode = True

    def __str__(self):
        return f"{self.query}," \
               f"{self.previous_chat}"


class FeatureModelVisualSimilarity(BaseModel):
    image_id: List[str] = Field(..., description="the array of relevant image id")
    query: str = Field(..., description="the query")

    class Config:
        orm_mode = True

    def __str__(self):
        return f"{self.query}," \
               f"{self.image_id}"
