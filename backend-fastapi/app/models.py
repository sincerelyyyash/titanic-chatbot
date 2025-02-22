from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    image: Optional[str] = None

