from pydantic import BaseModel
from typing import Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    visualization_data: Optional[Dict[str, Any]] = None

