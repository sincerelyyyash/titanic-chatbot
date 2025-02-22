from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.llm_agent import ask_titanic_ai
from app.data_analysis import generate_visual
from app.models import QueryRequest, QueryResponse

app = FastAPI(title="Titanic Chatbot API", version="1.0")

@app.post("/chat/", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        response_text = ask_titanic_ai(request.query)

        words = request.query.lower().split()
        image_data = None
        for keyword in ["age", "fare", "survival"]:
            if keyword in words:
                image_data = generate_visual(keyword)
                break

        return JSONResponse(content={"answer": response_text, "image": image_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

