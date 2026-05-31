import os
import shutil
import time
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_existing_vectorstore():
    # Kiểm tra xem DB đã tồn tại và có dữ liệu chưa
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_model())
    return None

def build_vectorstore(chunks, backup=False):
    time.sleep(1) # Nghỉ 1 giây để hệ điều hành nhả file lock của DB cũ
    
    # 1. Cơ chế Backup: Dời thư mục cũ sang tên mới
    if backup and os.path.exists(PERSIST_DIR):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{PERSIST_DIR}_backup_{timestamp}"
        try:
            shutil.move(PERSIST_DIR, backup_dir)
            print(f"=== Đã sao lưu Vector DB cũ sang: {backup_dir} ===")
        except Exception as e:
            print(f"Lỗi sao lưu: {e}")
            
    # 2. Cơ chế dọn dẹp: Xóa sạch thư mục hiện tại để chống trùng lặp (Duplicate)
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
        except Exception as e:
            pass

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    db = Chroma.from_texts(
        texts=texts,
        embedding=get_embedding_model(),
        metadatas=metadatas,
        persist_directory=PERSIST_DIR
    )

    return db
