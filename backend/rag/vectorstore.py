import os
import time

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "thu_tuc_hanh_chinh_lai_chau")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)


def get_embedding_model():
    """
    Tạo embedding model dùng chung cho build và search.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )


def get_qdrant_client():
    """
    Kết nối tới Qdrant local đang chạy qua Docker.
    """
    return QdrantClient(
        url=QDRANT_URL
    )


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """
    Kiểm tra collection đã tồn tại trong Qdrant hay chưa.
    """
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        return collection_name in collection_names
    except Exception as e:
        print(f"[QDRANT CHECK ERROR] {e}")
        return False


def load_existing_vectorstore():
    """
    Load collection Qdrant đã tồn tại.

    Hàm giữ nguyên tên cũ để main.py/admin.py/pipeline.py không cần đổi interface.
    """
    client = get_qdrant_client()

    if not collection_exists(client, COLLECTION_NAME):
        print(f"[QDRANT] Chưa có collection: {COLLECTION_NAME}")
        return None

    try:
        db = QdrantVectorStore.from_existing_collection(
            embedding=get_embedding_model(),
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
        )

        print(f"[QDRANT] Đã load collection: {COLLECTION_NAME}")
        return db

    except Exception as e:
        print(f"[QDRANT LOAD ERROR] {e}")
        return None


def build_vectorstore(chunks, backup=False):
    """
    Build lại Qdrant collection từ danh sách Document.

    Giai đoạn local: xóa collection cũ rồi tạo lại để tránh trùng dữ liệu.
    Tham số backup giữ lại để tương thích với admin.py.
    """
    time.sleep(1)

    client = get_qdrant_client()

    try:
        if collection_exists(client, COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"[QDRANT] Đã xóa collection cũ: {COLLECTION_NAME}")
    except Exception as e:
        print(f"[QDRANT DELETE WARNING] {e}")

    db = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )

    print(f"[QDRANT] Đã build collection: {COLLECTION_NAME}")
    print(f"[QDRANT] Tổng số chunks: {len(chunks)}")

    return db
