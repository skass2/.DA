import time
import json
from collections import Counter
from sentence_transformers import CrossEncoder
from rag.intent import handle_intent
from rag.normalizer import normalize_query
from rag.config import get_llm, get_fallback_llms  # Đảm bảo import factory của bạn

# ===== CONFIG =====
CACHE = {}

# Tải dữ liệu bổ trợ
try:
    with open("data/entities.json", "r", encoding="utf-8") as f:
        ENTITIES_DATA = json.load(f)
    with open("data/synonyms.json", "r", encoding="utf-8") as f:
        SYNONYMS_DATA = json.load(f)
except Exception as e:
    print(f"[FILE LOAD ERROR]: {e}")
    ENTITIES_DATA, SYNONYMS_DATA = {}, {}

# ===== KHỞI TẠO RERANKER (Chỉ load 1 lần duy nhất để tối ưu hiệu năng) =====
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("[RERANKER] Đã tải thành công CrossEncoder")
except Exception as e:
    print(f"[RERANKER INIT ERROR]: {e}")
    reranker = None
    
def detect_procedure_name(query: str, history=None):
    query = query.lower()
    
    # Ưu tiên 1: Nếu trong lịch sử gần nhất đã nhắc tới 1 thủ tục, hãy giữ lại nó
    if history:
        for msg in reversed(history[-2:]): # Xem 2 tin nhắn gần nhất
            content = msg['content'].lower()
            for proc_name in ENTITIES_DATA.keys():
                if proc_name.lower() in content:
                    return proc_name

    # Ưu tiên 2: Tìm kiếm từ khóa như cũ
    best_match = None
    max_score = 0
    for proc_name, keywords in ENTITIES_DATA.items():
        score = 0
        for k in keywords:
            if str(k).lower() in query:
                score += len(str(k).split())
        if score > max_score:
            max_score = score
            best_match = proc_name
            
    return best_match if max_score >= 2 else None

def detect_field(query: str):
    q = query.lower()
    field_map = {
        "phí": ["Phí", "Lệ phí"],
        "thời hạn": ["Thời hạn giải quyết"],
        "hồ sơ": ["Thành phần hồ sơ"],
        "cách thức": ["Cách thức thực hiện"],
        "cơ quan": ["Cơ quan thực hiện"],
        "pháp lý": ["Căn cứ pháp lý", "Cơ quan ban hành", "Cơ quan phối hợp"],
        "trình tự": ["Trình tự thực hiện"],
        "kết quả": ["Kết quả thực hiện"],
        "điều kiện": ["Yêu cầu điều kiện"],
        "đối tượng": ["Đối tượng thực hiện"]
    }
    for key, keywords in SYNONYMS_DATA.items():
        if any(k in q for k in keywords):
            return field_map.get(key)
    return None

# ===== HÀM THỰC THI LLM VỚI CƠ CHẾ FALLBACK =====
def smart_llm_invoke(prompt: str):
    """
    Thử gọi Model chính, nếu lỗi 429 hoặc lỗi kết nối 
    thì tự động duyệt qua danh sách Fallback.
    """
    primary = get_llm()
    fallbacks = get_fallback_llms()
    all_models = [primary] + fallbacks

    for model in all_models:
        if model is None:
            continue
        
        model_name = getattr(model, 'model', 'Ollama/OpenRouter')
        try:
            print(f"[*] Đang thử Model: {model_name}")
            res = model.invoke(prompt)
            return res.content.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[!] {model_name} hết hạn mức (429). Chuyển model...")
            else:
                print(f"[!] Lỗi {model_name}: {e}")
            continue # Thử model tiếp theo trong list
            
    return None

def ask_rag(db, query, session_id, history=None):
    if db is None:
        return "Hệ thống cơ sở dữ liệu hiện không khả dụng. Vui lòng thử lại sau."

    raw_query = query
    
    # 1. Intent xã giao
    intent_answer = handle_intent(raw_query)
    if intent_answer and len(raw_query.split()) < 5:
        return intent_answer

    # 2. Xử lý ngữ cảnh & Rewrite (Sử dụng Smart LLM)
    if history and len(history) > 0:
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history[-3:]])
        rewrite_prompt = f"""Bạn là một chuyên gia ngôn ngữ học. Dựa vào lịch sử hội thoại, hãy viết lại câu hỏi mới nhất của người dùng thành một câu hỏi duy nhất, độc lập và đầy đủ ý nghĩa.
TUYỆT ĐỐI KHÔNG TRẢ LỜI CÂU HỎI, KHÔNG GIẢI THÍCH GÌ THÊM. CHỈ TRẢ VỀ CÂU HỎI ĐÃ ĐƯỢC VIẾT LẠI.

LỊCH SỬ:
{history_str}
CÂU HỎI MỚI: {raw_query}

CÂU HỎI VIẾT LẠI:"""
        
        rewritten = smart_llm_invoke(rewrite_prompt)
        if rewritten:
            query = rewritten
            print(f"[REWRITE]: {query}")

    # 3. Chuẩn hóa & Cache
    query = normalize_query(query)
    query_key = query.lower().strip()
    if query_key in CACHE:
        print("[CACHE HIT]")
        return CACHE[query_key]

    # Xóa Cache nếu dung lượng quá lớn để tránh rò rỉ bộ nhớ (Memory Leak)
    if len(CACHE) > 1000:
        CACHE.clear()

    try:
        print(f"\n===== XỬ LÝ TRUY VẤN: {query} =====")
        
        # 4. Nhận diện mục tiêu
        # Thay đổi dòng này trong hàm ask_rag:
        detected_proc = detect_procedure_name(query, history=history)
        field = detect_field(raw_query)

        # 5. Retrieval (Hybrid)
        docs = []
        
        # 5.1 Tìm kiếm ngữ nghĩa tự do (Semantic Search)
        retriever_semantic = db.as_retriever(search_kwargs={"k": 15})
        docs_semantic = retriever_semantic.invoke(query)
        docs.extend(docs_semantic)

        # 5.2 Tìm kiếm theo bộ lọc Keyword (Nếu có)
        if detected_proc:
            print(f"[SYSTEM]: Keyword gợi ý thủ tục: {detected_proc}")
            retriever_filter = db.as_retriever(search_kwargs={"k": 10, "filter": {"name": detected_proc}})
            docs_filter = retriever_filter.invoke(query)
            docs.extend(docs_filter)
            
        # 5.3 Loại bỏ trùng lặp (Deduplicate)
        unique_docs = []
        seen_content = set()
        for d in docs:
            if d.page_content not in seen_content:
                unique_docs.append(d)
                seen_content.add(d.page_content)
        docs = unique_docs

        if not docs:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

        # 6. Rerank
        if reranker:
            pairs = [(query, d.page_content) for d in docs]
            scores = reranker.predict(pairs)
            for i, d in enumerate(docs):
                d.metadata['score'] = scores[i]
            docs = sorted(docs, key=lambda x: x.metadata['score'], reverse=True)
        else:
            # Fallback nếu không có reranker (chạy tạm không chấm điểm)
            for d in docs:
                d.metadata['score'] = 1.0

        # 7. Chọn thủ tục thắng cuộc bằng Reranker
        proc_scores = {}
        # Lấy top 5 chunk có điểm cao nhất để bầu chọn thủ tục chính xác nhất
        for d in docs[:5]:
            p_name = d.metadata.get("name")
            if p_name:
                proc_scores[p_name] = proc_scores.get(p_name, 0) + d.metadata['score']
                
        main_name = max(proc_scores, key=proc_scores.get) if proc_scores else "Thủ tục không xác định"

        print(f"[WINNER]: {main_name}")

        # 8. Lọc Chunk theo Winner và Field
        final_docs = [d for d in docs if d.metadata.get("name") == main_name]
        if field:
            # field lúc này là một list các trường tương ứng (Ví dụ: ["Phí", "Lệ phí"])
            field_docs = [d for d in final_docs if d.metadata.get("field") in field]
            if field_docs:
                final_docs = field_docs
                print(f"[FIELD FILTER]: Đã lọc theo mục {field}")

        # 9. Tổng hợp Context
        seen = set()
        context_chunks = []
        for d in final_docs:
            if d.page_content not in seen:
                context_chunks.append(d.page_content)
                seen.add(d.page_content)
                # Chỉ lấy tối đa 5 chunks tốt nhất (sau Rerank) để LLM không bị ngợp và ảo giác
                if len(context_chunks) >= 5:
                    break
        
        context = "\n\n".join(context_chunks)
        print(f"Context nè: {context}")
        
        # 10. Generate (Sử dụng Smart LLM với Fallback)
        prompt = f"""Bạn là công chức tiếp nhận hồ sơ. CHỈ trả lời dựa trên CONTEXT.

YÊU CẦU CỰC ĐOAN:
1. KHÔNG được nói "liên hệ cơ quan chức năng" nếu trong CONTEXT đã có quy định.
2. KHÔNG được trả lời chung chung. Nếu hỏi về hồ sơ, phải liệt kê rõ Tên giấy tờ, Số lượng.
3. Nếu CONTEXT có các thông số kỹ thuật (khoảng cách, mét, ngày, lý trình), BẮT BUỘC phải đưa vào câu trả lời.
4. Nếu KHÔNG thấy thông tin trong CONTEXT, chỉ nói: "Dữ liệu hiện tại không đề cập". Tuyệt đối không đoán.
5. TUYỆT ĐỐI KHÔNG dùng các cụm từ như "Theo CONTEXT cung cấp", "Dựa vào tài liệu". Hãy trả lời tự nhiên như một người tư vấn.
6. TRẢ LỜI NGẮN GỌN, TRỰC TIẾP VÀO TRỌNG TÂM CÂU HỎI. KHÔNG tự ý suy diễn hoặc bịa thêm thông tin ngoài CONTEXT.
7. BẮT BUỘC TRẢ LỜI BẰNG TIẾNG VIỆT.
8. TRÌNH BÀY RÕ RÀNG: Khi liệt kê các bước, hồ sơ hoặc danh sách, BẮT BUỘC PHẢI XUỐNG DÒNG CHO MỖI GẠCH ĐẦU DÒNG.

CONTEXT:
{context}

CÂU HỎI: {query}
TRẢ LỜI:"""

        answer = smart_llm_invoke(prompt)

        if answer:
            CACHE[query_key] = answer
            return answer
        else:
            return "Hiện tại tất cả dịch vụ AI (Gemini, Ollama) đều không phản hồi. Vui lòng thử lại sau."

    except Exception as e:
        print(f"[CRITICAL ERROR]: {e}")
        return "Hệ thống đang bận, vui lòng thử lại sau."