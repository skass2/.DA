from fastapi import FastAPI
from rag.loader import load_data
from rag.chunker import create_chunks
from rag.vectorstore import build_vectorstore
from rag.pipeline import ask_rag
from rag.config import get_llm, get_fallback_llm
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# GLOBAL
db = None
llm = None
fallback_llm = None


@app.on_event("startup")
def startup():
    global db, llm, fallback_llm

    print("=== STARTUP ===")

    data = load_data()
    print("Loaded data:", len(data))

    chunks = create_chunks(data)
    print("Chunks:", len(chunks))

    db = build_vectorstore(chunks)

    llm = get_llm()
    fallback_llm = get_fallback_llm()

    print("=== READY ===")


@app.get("/")
def root():
    return {"status": "OK"}


@app.get("/chat")
def chat(q: str):
    answer = ask_rag(db, q, llm, fallback_llm)
    return {"answer": answer}
