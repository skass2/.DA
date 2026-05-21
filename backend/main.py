from fastapi import FastAPI, Request
from dotenv import load_dotenv
from rag.loader import load_data
from rag.chunker import create_chunks
from rag.vectorstore import build_vectorstore, load_existing_vectorstore
from rag.pipeline import ask_rag
from rag.config import get_llm, get_lightweight_llm, get_fallback_llms
from fastapi.middleware.cors import CORSMiddleware
from rag.memory import get_history, save_message
from routers import user, admin, auth
import os
import firebase_admin
from firebase_admin import credentials

# ===== LOAD ENV =====
load_dotenv(dotenv_path=".env")
# Bảo mật: Chỉ in 4 ký tự cuối của API Key thay vì toàn bộ
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API KEY LOADED: ...{api_key[-4:] if api_key else 'NOT FOUND'}")

# ===== FIREBASE INIT =====
# Khởi tạo Firebase Admin SDK với Service Account Key
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        print("=== FIREBASE INITIALIZED ===")
except Exception as e:
    print(f"=== FIREBASE INIT ERROR: {e} ===")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== STARTUP =====
@app.on_event("startup")
def startup():
    print("=== STARTING RAG SYSTEM ===")
    try:
        # Ưu tiên load DB đã có trên ổ cứng để khởi động nhanh
        db = load_existing_vectorstore()
        if db is None:
            print("=== KHÔNG TÌM THẤY VECTOR DB. TIẾN HÀNH TẠO MỚI... ===")
            data = load_data()
            chunks = create_chunks(data)
            app.state.db = build_vectorstore(chunks, backup=False)
        else:
            print("=== ĐÃ LOAD VECTOR DB CÓ SẴN (BỎ QUA CHUNKING) ===")
            app.state.db = db

        app.state.llm = get_llm()
        app.state.lightweight_llm = get_lightweight_llm()
        app.state.fallback_llms = get_fallback_llms()
        print("=== SYSTEM READY ===")
    except Exception as e:
        print(f"=== STARTUP ERROR: {e} ===")
        app.state.db = None

app.include_router(user.router)
app.include_router(admin.router)
app.include_router(auth.router)

#       py -m uvicorn main:app --reload --port 8000