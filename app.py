from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_system import RagSystem

app = FastAPI()

rag = RagSystem()
rag.initialize()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    return rag.ask(request.question)

app.mount("/", StaticFiles(directory="static", html=True), name="static")