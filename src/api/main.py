from fastapi import FastAPI
from pydantic import BaseModel
from src.core.langgraph_flow import query_rag_system

app = FastAPI(title="Medical RAG Backend API")


class ChatRequest(BaseModel):
    question: str


@app.post("/api/v1/chat")
async def chat_endpoint(payload: ChatRequest):
    result = query_rag_system(payload.question)
    # query_rag_system() trả về GraphState đầy đủ (intent, entities,
    # raw_documents, filtered_documents, question_vector...) — chỉ trả
    # đúng phần app_gui.py cần (key "answer") để tránh KeyError phía
    # frontend, đồng thời tránh gửi thừa question_vector (384 số float)
    # và list Document object qua network mỗi lần chat.
    return {"answer": result.get("generation", "")}