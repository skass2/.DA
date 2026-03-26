from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore(chunks):
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory="chroma_db"
    )

    return db
