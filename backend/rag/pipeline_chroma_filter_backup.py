import time
import json
import re
from collections import OrderedDict
from sentence_transformers import CrossEncoder
from rag.intent import handle_intent
from rag.normalizer import normalize_query
from rag.config import get_llm, get_lightweight_llm, get_fallback_llms

# ===== CONFIG =====
CACHE = OrderedDict()
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 60 * 60
HISTORY_MESSAGES_FOR_REWRITE = 6
MAX_CONTEXT_CHUNKS = 10
LLM_RETRY_ATTEMPTS = 2
LLM_RETRY_BASE_DELAY = 1.5

SYSTEM_PROMPT = """Bạn là chuyên viên tư vấn thủ tục hành chính nhiệt tình, thấu hiểu và CÓ TƯ DUY PHẢN BIỆN SẮC BÉN. Hãy kết hợp thông tin từ CONTEXT và LỊCH SỬ HỘI THOẠI để tư vấn cho người dùng.

YÊU CẦU BẮT BUỘC:
1. CHỈ SỬ DỤNG THÔNG TIN CÓ TRONG CONTEXT. TUYỆT ĐỐI KHÔNG TỰ BỊA ĐẶT, suy diễn hoặc tự ý thêm tên các văn bản pháp luật, thông tư, nghị định không có trong CONTEXT.
2. BÓC TÁCH VÀ PHẢN BIỆN LOGIC (RẤT QUAN TRỌNG): Nếu người dùng đưa ra các giả định sai lệch, tự ý gài bẫy bằng cách lấy "ngày ban hành văn bản pháp lý" (như ngày, tháng, năm của Luật/Thông tư) để cộng trừ nhân chia làm "thời gian giải quyết", bạn BẮT BUỘC phải chỉ ra sự vô lý này và đính chính lại bằng thời gian giải quyết thực tế có trong CONTEXT.
3. XÁC ĐỊNH ĐÚNG TRỌNG TÂM: Nếu người dùng đề cập đến nhiều thông tin nhiễu, hãy nhận diện mục đích chính (ví dụ: làm thủ tục gì) để tư vấn. Bác bỏ các thông tin không liên quan.
4. Trả lời đúng trọng tâm. Nếu hỏi về hồ sơ, phải liệt kê rõ Tên giấy tờ, Số lượng.
5. CÓ THỂ TƯ DUY, TÍNH TOÁN, ĐỒNG CẢM: Nếu người dùng bức xúc về tính toán thời hạn hợp lý, hãy đối chiếu với số ngày giải quyết trong CONTEXT để giải thích và xoa dịu họ.
6. Nếu KHÔNG thấy thông tin trong CONTEXT và LỊCH SỬ, hãy lịch sự thông báo "Dữ liệu hiện tại chưa có thông tin quy định về vấn đề này".
7. TUYỆT ĐỐI KHÔNG dùng các cụm từ như "Theo CONTEXT cung cấp", "Dựa vào tài liệu".
8. BẮT BUỘC TRẢ LỜI BẰNG TIẾNG VIỆT.
9. TRÌNH BÀY RÕ RÀNG: Khi liệt kê các bước, hồ sơ, danh sách, BẮT BUỘC PHẢI XUỐNG DÒNG. TUYỆT ĐỐI KHÔNG SỬ DỤNG KÝ TỰ MARKDOWN (như **, *) để in đậm, tránh gây lỗi hiển thị.
"""

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
    
def detect_procedure_name(query: str, history=None, raw_query: str = ""):
    query_lower = query.lower()
    raw_query_lower = raw_query.lower()
    
    # Ưu tiên 0: Tìm chính xác tên thủ tục trong câu hỏi gốc hoặc câu hỏi đã rewrite
    for proc_name in ENTITIES_DATA.keys():
        proc_name_lower = proc_name.lower().strip()
        if proc_name_lower and (proc_name_lower in raw_query_lower or proc_name_lower in query_lower):
            return proc_name

    # Ưu tiên 1: Nếu trong lịch sử gần nhất đã nhắc tới 1 thủ tục, hãy giữ lại nó
    if history:
        for msg in reversed(history[-2:]): # Xem 2 tin nhắn gần nhất
            content = msg['content'].lower()
            for proc_name in ENTITIES_DATA.keys():
                proc_name_lower = proc_name.lower().strip()
                if proc_name_lower and proc_name_lower in content:
                    return proc_name

    # Ưu tiên 2: Tìm kiếm từ khóa như cũ
    best_match = None
    max_score = 0
    for proc_name, keywords in ENTITIES_DATA.items():
        score = 0
        for k in keywords:
            k_str = str(k).lower().strip()
            # Dùng regex \b để tránh bắt nhầm một phần của từ (ví dụ: 'tỉnh' trong 'tỉnh táo')
            if re.search(r'\b' + re.escape(k_str) + r'\b', query_lower):
                score += len(k_str.split())
        if score > max_score:
            max_score = score
            best_match = proc_name
            
    # Tăng độ khó lên 3 để tránh sập bẫy các từ khóa chung chung như "cấp tỉnh" (2 âm tiết)
    return best_match if max_score >= 3 else None

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
    detected_fields = []
    for key, keywords in SYNONYMS_DATA.items():
        if any(k in q for k in keywords):
            mapped = field_map.get(key)
            if mapped:
                detected_fields.extend(mapped)
    return detected_fields if detected_fields else None

# ===== HELPERS =====
def estimate_tokens(text: str) -> int:
    """Ước lượng token đơn giản để log và tối ưu prompt sớm."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _cache_get(query_key: str):
    cached = CACHE.get(query_key)
    if not cached:
        return None

    created_at, answer = cached
    if time.time() - created_at > CACHE_TTL_SECONDS:
        CACHE.pop(query_key, None)
        return None

    CACHE.move_to_end(query_key)
    print("[CACHE HIT]")
    return answer


def _cache_set(query_key: str, answer: str):
    CACHE[query_key] = (time.time(), answer)
    CACHE.move_to_end(query_key)

    while len(CACHE) > CACHE_MAX_SIZE:
        CACHE.popitem(last=False)


def _is_rate_limit_error(error: Exception) -> bool:
    error_text = str(error)
    return any(marker in error_text for marker in ["429", "RESOURCE_EXHAUSTED", "TooManyRequests", "rate limit"])


def _model_name(model) -> str:
    return getattr(model, "model", None) or getattr(model, "model_name", None) or model.__class__.__name__


# ===== HÀM THỰC THI LLM VỚI CƠ CHẾ FALLBACK =====
def smart_llm_invoke(prompt: str, prefer_lightweight: bool = False):
    """
    Gọi LLM với primary Gemini 2.5 Flash, dùng Flash Lite cho tác vụ nhẹ/fallback,
    có retry + delay khi gặp 429 để giảm lỗi quota tạm thời.
    """
    primary = get_llm()
    lightweight = get_lightweight_llm()
    fallbacks = get_fallback_llms()

    if prefer_lightweight:
        candidates = [lightweight, primary] + fallbacks
    else:
        candidates = [primary] + fallbacks

    all_models = []
    seen = set()
    for model in candidates:
        if model is None:
            continue
        model_id = id(model)
        if model_id in seen:
            continue
        seen.add(model_id)
        all_models.append(model)

    prompt_tokens = estimate_tokens(prompt)

    for model in all_models:
        model_name = _model_name(model)

        for attempt in range(1, LLM_RETRY_ATTEMPTS + 1):
            started_at = time.perf_counter()
            try:
                print(f"[*] Đang thử Model: {model_name} | attempt={attempt} | prompt_tokens≈{prompt_tokens}")
                res = model.invoke(prompt)
                content = res.content.strip()
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                print(
                    f"[LLM OK] model={model_name} attempt={attempt} "
                    f"response_tokens≈{estimate_tokens(content)} latency_ms={elapsed_ms}"
                )
                return content
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                if _is_rate_limit_error(e):
                    delay = LLM_RETRY_BASE_DELAY * attempt
                    print(
                        f"[LLM 429] model={model_name} attempt={attempt} "
                        f"latency_ms={elapsed_ms} delay_s={delay:.1f}"
                    )
                    time.sleep(delay)
                    continue

                print(f"[LLM ERROR] model={model_name} attempt={attempt} latency_ms={elapsed_ms}: {e}")
                break

    return None


def ask_rag(db, query, session_id, history=None):
    request_started_at = time.perf_counter()

    if db is None:
        return "Hệ thống cơ sở dữ liệu hiện không khả dụng. Vui lòng thử lại sau."

    raw_query = query
    
    # 1. Intent xã giao
    intent_answer = handle_intent(raw_query)
    if intent_answer and len(raw_query.split()) < 5:
        return intent_answer

    # 2. Xử lý ngữ cảnh & Rewrite (Trích xuất thông tin cốt lõi, loại bỏ nhiễu)
    if history and len(history) > 0:
        history_formatted = "LỊCH SỬ HỘI THOẠI:\n" + "\n".join([f"{'Người dùng' if h['role'] == 'user' else 'Chuyên viên'}: {h['content']}" for h in history[-HISTORY_MESSAGES_FOR_REWRITE:]])
    else:
        history_formatted = "LỊCH SỬ HỘI THOẠI:\n(Chưa có)"

    rewrite_prompt = f"""Bạn là hệ thống trích xuất từ khóa tìm kiếm (Search Engine Optimizer). Dựa vào lịch sử hội thoại và câu hỏi mới nhất, hãy tạo ra một câu truy vấn NGẮN GỌN, CHÍNH XÁC NHẤT để tìm kiếm tài liệu.
BẮT BUỘC:
1. GIỮ NGUYÊN TÊN THỦ TỤC HÀNH CHÍNH (nếu có).
2. GIỮ NGUYÊN CÁC CÂU HỎI PHỤ, chi tiết quan trọng người dùng muốn biết (ví dụ: cần giấy tờ gì, nộp ở đâu, mất bao lâu, đúng không...).
3. LOẠI BỎ HOÀN TOÀN các từ xưng hô, kể lể hoàn cảnh cá nhân, địa danh không liên quan (ví dụ: thằng con trai tôi, ở Hà Nội, ông bạn già, hôm qua, định mở văn phòng...).
4. CHỈ TRẢ VỀ CÂU TRUY VẤN ĐÃ RÚT GỌN, KHÔNG GIẢI THÍCH GÌ THÊM.

{history_formatted}
CÂU HỎI MỚI: {raw_query}

CÂU TRUY VẤN TỐI ƯU:"""
    
    rewritten = smart_llm_invoke(rewrite_prompt, prefer_lightweight=True)
    if rewritten:
        query = rewritten
        print(f"[REWRITE]: {query}")

    llm_query = query # Giữ lại câu hỏi trước khi bị chuẩn hóa (biến dạng) để dùng cho Rerank

    # 3. Chuẩn hóa & Cache
    query = normalize_query(query)
    query_key = query.lower().strip()
    cached_answer = _cache_get(query_key)
    if cached_answer:
        return cached_answer

    try:
        print(f"\n===== XỬ LÝ TRUY VẤN: {query} =====")
        
        # 4. Nhận diện mục tiêu
        # Thay đổi dòng này trong hàm ask_rag để truyền thêm raw_query:
        detected_proc = detect_procedure_name(query, history=history, raw_query=raw_query)
        field = detect_field(raw_query)

        # 5. Retrieval (Hybrid)
        docs = []
        
        # 5.1 Tìm kiếm ngữ nghĩa tự do (Semantic Search) - Tăng k lên 20 để không lọt lưới do nhiễu
        retriever_semantic = db.as_retriever(search_kwargs={"k": 20})
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
            pairs = [(llm_query, d.page_content) for d in docs]
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
        # Lấy top 8 chunk có điểm cao nhất để bầu chọn (mở rộng phễu để bắt các thủ tục bị nhiễu đẩy xuống dưới)
        for d in docs[:8]:
            p_name = d.metadata.get("name")
            if p_name:
                # Dùng max() thay vì cộng dồn (sum) vì điểm CrossEncoder có thể bị âm (không liên quan)
                if p_name not in proc_scores:
                    proc_scores[p_name] = d.metadata['score']
                else:
                    proc_scores[p_name] = max(proc_scores[p_name], d.metadata['score'])
                
        # Lấy top 2 thủ tục có điểm cao nhất thay vì chỉ 1 để tránh đánh rơi ngữ cảnh khi bị nhiễu
        top_procs = sorted(proc_scores.items(), key=lambda item: item[1], reverse=True)[:2]
        allowed_proc_names = [p[0] for p in top_procs] if top_procs else ["Thủ tục không xác định"]

        print(f"[WINNER]: {allowed_proc_names}")

        # [PARENT-CHILD RETRIEVER] Bổ sung chunk "Cách thức thực hiện" của các Winner 
        # để không bỏ sót các lưu ý quan trọng về thời gian, chi phí nằm ở cuối văn bản.
        for p_name in allowed_proc_names:
            if p_name != "Thủ tục không xác định":
                extra_docs = db.similarity_search(query, k=3, filter={"$and": [{"name": p_name}, {"section_type": "method"}]})
                for ed in extra_docs:
                    if ed.page_content not in seen_content:
                        ed.metadata['score'] = 0.99  # Gán điểm cao để không bị loại bỏ
                        # Đánh lừa bộ lọc Field để chunk này luôn được ưu tiên bám sát theo Field gốc
                        if field and isinstance(field, list):
                            ed.metadata['field'] = field[0]
                        docs.insert(0, ed)
                        seen_content.add(ed.page_content)

        # 8. Lọc Chunk theo Winner và Field
        final_docs = [d for d in docs if d.metadata.get("name") in allowed_proc_names]
        if field:
            # field lúc này là một list các trường tương ứng (Ví dụ: ["Phí", "Lệ phí"])
            field_docs = [d for d in final_docs if d.metadata.get("field") in field]
            if field_docs:
                other_docs = [d for d in final_docs if d not in field_docs]
                final_docs = field_docs + other_docs
                print(f"[FIELD FILTER]: Đã ưu tiên các mục {field} lên đầu")

        # 9. Tổng hợp Context
        seen = set()
        context_chunks = []
        for d in final_docs:
            if d.page_content not in seen:
                context_chunks.append(d.page_content)
                seen.add(d.page_content)
                # Chỉ lấy tối đa các chunks tốt nhất (sau Rerank) để LLM không bị ngợp và ảo giác
                if len(context_chunks) >= MAX_CONTEXT_CHUNKS:
                    break
        
        context = "\n\n".join(context_chunks)
        print(
            f"[CONTEXT] chunks={len(context_chunks)} chars={len(context)} "
            f"tokens≈{estimate_tokens(context)}"
        )
        
        # 10. Generate (Sử dụng system prompt cố định + Smart LLM với Fallback)
        prompt = f"""{SYSTEM_PROMPT}

{history_formatted}

CONTEXT TÀI LIỆU (Cập nhật mới nhất):
{context}

CÂU HỎI HIỆN TẠI: {raw_query}
TRẢ LỜI:"""

        answer = smart_llm_invoke(prompt)

        if answer:
            # Hậu xử lý: Xóa các ký tự markdown in đậm (**)
            answer = answer.replace('**', '')
            # Chuyển đổi list dạng '*' thành '-'
            answer = re.sub(r'(?m)^\s*\*\s+', '- ', answer)
            # Rút gọn khoảng trống dư thừa (dồn 3+ dấu xuống dòng thành 2)
            answer = re.sub(r'\n{3,}', '\n\n', answer).strip()

            _cache_set(query_key, answer)
            elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
            print(
                f"[RAG OK] session_id={session_id} latency_ms={elapsed_ms} "
                f"query_tokens≈{estimate_tokens(raw_query)} answer_tokens≈{estimate_tokens(answer)}"
            )
            return answer
        else:
            return "Hiện tại tất cả dịch vụ AI (Gemini, Ollama) đều không phản hồi. Vui lòng thử lại sau."

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
        print(f"[CRITICAL ERROR] latency_ms={elapsed_ms}: {e}")
        return "Hệ thống đang bận, vui lòng thử lại sau."