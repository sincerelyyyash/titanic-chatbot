from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.llm_agent import ask_titanic_ai
from app.data_analysis import generate_visualization_data
from app.models import QueryRequest, QueryResponse

app = FastAPI(title="Titanic Chatbot API", version="1.0")

@app.post("/chat/", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        response_text = ask_titanic_ai(request.query)

        words = request.query.lower().split()
        visualization_data = None
        for keyword in ["age", "fare", "survival", "gender", "class"]:
            if keyword in words:
                visualization_data = generate_visualization_data(request.query)
                break

        return JSONResponse(content={"answer": response_text, "visualization_data": visualization_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

