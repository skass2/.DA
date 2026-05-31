from fastapi import APIRouter, Query
from rag.vectorstore import load_existing_vectorstore
from rag.pipeline import ask_rag

router = APIRouter(prefix="/dev", tags=["Dev"])

_dev_db = None


def get_dev_db():
    """
    Load Qdrant collection một lần để test local.
    Route này chỉ dùng trong môi trường phát triển.
    """
    global _dev_db

    if _dev_db is None:
        _dev_db = load_existing_vectorstore()

    return _dev_db


@router.get("/chat")
def dev_chat(
    q: str = Query(..., description="Câu hỏi cần test"),
    session_id: str = Query("dev_qdrant_test", description="Session test local")
):
    """
    Endpoint test RAG không cần đăng nhập.

    Chỉ dùng local để kiểm tra Qdrant + pipeline.
    Không dùng endpoint này khi deploy thật.
    """
    db = get_dev_db()

    if db is None:
        return {
            "ok": False,
            "error": "Chưa load được Qdrant collection."
        }

    answer = ask_rag(
        db=db,
        query=q,
        session_id=session_id,
        history=[]
    )

    return {
        "ok": True,
        "query": q,
        "session_id": session_id,
        "answer": answer
    }