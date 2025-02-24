from pydantic import BaseModel
from typing import Dict, Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    visualization_data: Optional[Dict] = None

