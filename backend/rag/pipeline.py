import time
from typing import List
from sentence_transformers import CrossEncoder
# ====== CONFIG ======
MAX_CONTEXT_LENGTH = 5000
RETRY_TIMES = 3
RETRY_DELAY = 2


# ====== HELPER: RETRY ======
def call_llm_with_retry(llm, prompt: str, retries: int = RETRY_TIMES):
    for i in range(retries):
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"[LLM ERROR] Retry {i+1}/{retries}: {e}")
            time.sleep(RETRY_DELAY)

    return None


# ====== HELPER: BUILD CONTEXT ======
def build_context(docs, field=None):
    if not docs:
        return ""

    contexts = []

    for doc in docs:
        # ===== lọc đúng field =====
        if field:
            if doc.metadata.get("field") == field:
                contexts = list(set(contexts))
        else:
            contexts = list(set(contexts))

    context = "\n\n".join(contexts)

    return context[:MAX_CONTEXT_LENGTH]


# ====== HELPER: BUILD PROMPT ======
def build_prompt(query: str, context: str):
    return f"""
Bạn là chatbot tra cứu thủ tục hành chính.

YÊU CẦU:
- Trả lời NGẮN GỌN, ĐÚNG TRỌNG TÂM
- Nếu câu hỏi hỏi về 1 mục cụ thể (ví dụ: phí, hồ sơ, thời hạn) → chỉ trả lời đúng mục đó
- Không lan man
- CHỈ được dùng thông tin trong CONTEXT
- KHÔNG được suy đoán ngoài
- Nếu không có → trả lời: "Không tìm thấy thông tin"

---------------------
CONTEXT:
{context}
---------------------

CÂU HỎI:
{query}

TRẢ LỜI:
"""


# ====== RULE: SIMPLE QUESTION ======
KEYWORDS_SIMPLE = [
    "phí", "lệ phí",
    "thời hạn", "bao lâu",
    "ở đâu", "cơ quan",
    "cần gì", "hồ sơ"
]

def is_simple_question(query):
    return any(k in query.lower() for k in KEYWORDS_SIMPLE)


# ====== DETECT FIELD ======
def detect_field(query: str):
    query = query.lower()

    if "lệ phí" in query:
        return "Lệ phí"
    if "phí" in query:
        return "Phí"
    if "thời hạn" in query:
        return "Thời hạn giải quyết"
    if "hồ sơ" in query:
        return "Thành phần hồ sơ"
    if "trình tự" in query:
        return "Trình tự thực hiện"
    if "cơ quan" in query:
        return "Cơ quan thực hiện"

    return None


# ====== MAIN RAG FUNCTION ======
def ask_rag(db, query: str, llm, fallback_llm=None):
    context = ""

    try:
        print("\n===== NEW QUERY =====")
        print("Query:", query)

        # ===== detect field =====
        field = detect_field(query)
        print("Detected field:", field)

        # ===== retriever =====
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "filter": {"field": field} if field else {}
            }
        )

        docs = retriever.invoke(query)
        if not docs:
            retriever = db.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(query)

        print(f"Retrieved {len(docs)} documents")

        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]

        # ===== build context =====
        context = build_context(docs, field)

        print("Context preview:", context[:500])

        if not context.strip():
            return "Không tìm thấy thông tin phù hợp."

        # ===== RULE: SIMPLE → KHÔNG GỌI LLM =====
        if is_simple_question(query):
            print("[INFO] Simple question → skip LLM")
            return context[:500]

        # ===== build prompt =====
        prompt = build_prompt(query, context)
        print("Prompt length:", len(prompt))

        # ===== call LLM =====
        response = call_llm_with_retry(llm, prompt)

        if response and hasattr(response, "content"):
            return response.content

        raise Exception("Primary LLM failed")

    except Exception as e:
        print("[RAG ERROR]:", str(e))

        # ===== fallback KHÔNG GỌI LLM =====
        if context:
            print("[INFO] Fallback → dùng context")
            return f"Thông tin:\n{context[:300]}"
        return "Hệ thống đang bận, vui lòng thử lại sau."
