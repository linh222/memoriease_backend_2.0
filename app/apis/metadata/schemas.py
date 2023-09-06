from pydantic import BaseModel, Field
from typing import Optional


class ResponseModel(BaseModel):
    timestamp: str = Field(..., description="timestamp of submission")
    metadata: str = Field(..., description="the answer of the submission")
    response: Optional[str] = Field(..., description="true of false or null")
    topic: Optional[str] = Field(..., description="topic of the question")

    class Config:
        orm_mode = True
