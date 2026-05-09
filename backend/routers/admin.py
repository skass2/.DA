from fastapi import APIRouter, Request, Depends
from routers.auth import get_admin_user
from rag.loader import load_data
from rag.chunker import create_chunks
from rag.vectorstore import build_vectorstore

router = APIRouter(prefix="/admin", tags=["Admin"])

# API Admin: Khởi tạo lại ChromaDB nếu file JSON bị thay đổi
@router.post("/reload-vectordb")
def reload_vectordb(request: Request, current_admin: dict = Depends(get_admin_user)):
    try:
        data = load_data()
        chunks = create_chunks(data)
        request.app.state.db = build_vectorstore(chunks)
        return {"status": "success", "message": "Vector DB đã được cập nhật lại thành công!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}