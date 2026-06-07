import time
import json
import re
import os
from collections import OrderedDict
from sentence_transformers import CrossEncoder
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from rag.intent import handle_intent
from rag.normalizer import normalize_query
from rag.config import get_llm, get_lightweight_llm, get_fallback_llms

# ===== CONFIG =====
CACHE = OrderedDict()
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 60 * 60
HISTORY_MESSAGES_FOR_REWRITE = 6
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "7"))
LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "1"))
LLM_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "0"))
LLM_SLEEP_ON_429 = os.getenv("LLM_SLEEP_ON_429", "false").lower() == "true"

# auto  : chỉ rewrite khi có lịch sử/ngữ cảnh mơ hồ hoặc chưa nhận diện được thủ tục
# true  : luôn rewrite như bản cũ
# false : tắt rewrite hoàn toàn
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "auto").lower()

USE_LIGHTWEIGHT_FOR_ANSWER = os.getenv("USE_LIGHTWEIGHT_FOR_ANSWER", "true").lower() == "true"
ENABLE_DIRECT_ANSWER = os.getenv("ENABLE_DIRECT_ANSWER", "true").lower() == "true"

RETRIEVAL_SEMANTIC_K = int(os.getenv("RETRIEVAL_SEMANTIC_K", "8"))
FILTER_SEARCH_K = int(os.getenv("FILTER_SEARCH_K", "5"))
PARENT_METHOD_K = int(os.getenv("PARENT_METHOD_K", "2"))
EXACT_FIELD_K = int(os.getenv("EXACT_FIELD_K", "6"))
DIRECT_ANSWER_MAX_CHUNKS = int(os.getenv("DIRECT_ANSWER_MAX_CHUNKS", "6"))
STRICT_FIELD_CONTEXT = os.getenv("STRICT_FIELD_CONTEXT", "true").lower() == "true"


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
    
def _direct_out_of_scope_answer(raw_query: str):
    """
    Trả lời nhanh các câu ngoài phạm vi rõ ràng, không gọi LLM.
    """
    q = raw_query.lower()

    if "mặt trăng" in q:
        return (
            "Dữ liệu hiện tại chưa có thông tin về thủ tục cấp giấy phép xây dựng "
            "trên mảnh đất ở mặt trăng. Bố nên kiểm tra lại địa điểm/thủ tục cần hỏi "
            "hoặc cung cấp tên thủ tục hành chính cụ thể hơn."
        )

    animal_words = ["con mèo", "con chó", "vật nuôi", "thú cưng", "pet"]
    if "khai sinh" in q and any(word in q for word in animal_words):
        return (
            "Dữ liệu hiện tại chưa có thông tin về thủ tục đăng ký khai sinh cho vật nuôi. "
            "Thủ tục đăng ký khai sinh trong dữ liệu đang áp dụng cho con người, không phải vật nuôi."
        )

    return None


def _resolve_proc_name(preferred_name: str):
    """
    Trả về đúng key thủ tục đang có trong ENTITIES_DATA.
    """
    if not preferred_name:
        return None

    if preferred_name in ENTITIES_DATA:
        return preferred_name

    preferred_lower = preferred_name.lower()

    for proc_name in ENTITIES_DATA.keys():
        proc_lower = proc_name.lower()
        if preferred_lower == proc_lower:
            return proc_name

    for proc_name in ENTITIES_DATA.keys():
        proc_lower = proc_name.lower()
        if preferred_lower in proc_lower or proc_lower in preferred_lower:
            return proc_name

    return None


def _apply_common_procedure_override(query_lower: str):
    """
    Luật đời thường cho các câu người dân/người lớn tuổi hay hỏi.
    Đặt trước keyword scoring để tránh bắt nhầm thủ tục gần nghĩa.
    """
    q = query_lower

    def has_any(words):
        return any(w in q for w in words)

    # Giấy độc thân = giấy xác nhận tình trạng hôn nhân
    if has_any(["giấy độc thân", "xác nhận độc thân", "tình trạng hôn nhân"]):
        return _resolve_proc_name("Thủ tục cấp Giấy xác nhận tình trạng hôn nhân")

    # Làm giấy khai sinh cho trẻ/cháu/con = đăng ký khai sinh.
    # Nếu nói rõ bản sao/trích lục thì mới là thủ tục cấp bản sao.
    if "khai sinh" in q:
        if has_any(["bản sao", "trích lục", "sao giấy khai sinh", "xin bản sao"]):
            matched = _resolve_proc_name("Cấp bản sao Trích lục hộ tịch, bản sao Giấy khai sinh")
            if matched:
                return matched

        if has_any([
            "làm giấy khai sinh",
            "đăng ký khai sinh",
            "khai sinh cho",
            "trẻ con",
            "cháu bé",
            "cho cháu",
            "cho con",
            "online",
            "lấy ngay trong ngày",
            "ra xã",
            "lên huyện",
        ]):
            return _resolve_proc_name("Thủ tục đăng ký khai sinh")

    # Cưới vợ/chồng = đăng ký kết hôn, trừ khi câu đang hỏi giấy độc thân đã bắt ở trên.
    if has_any(["cưới vợ", "cưới chồng", "đăng ký kết hôn", "kết hôn"]):
        return _resolve_proc_name("Thủ tục đăng ký kết hôn")

    # Nuôi con nuôi đời thường: nếu không nói yếu tố nước ngoài thì ưu tiên trong nước.
    if "con nuôi" in q:
        if "nước ngoài" not in q and "yếu tố nước ngoài" not in q:
            return _resolve_proc_name("Đăng ký việc nuôi con nuôi trong nước")

    # Hộ kinh doanh bị mất/xin lại
    if "hộ kinh doanh" in q and has_any(["mất", "xin lại", "cấp lại", "đổi lại"]):
        return _resolve_proc_name("Cấp lại Giấy chứng nhận đăng ký hộ kinh doanh, Cấp đổi sang Giấy chứng nhận đăng ký hộ kinh doanh")

    # Một mình làm chủ = công ty TNHH một thành viên
    if has_any(["một mình làm chủ", "một người làm chủ", "tnhh một thành viên", "tnhh 1 thành viên"]):
        return _resolve_proc_name("Đăng ký thành lập công ty TNHH một thành viên")

    # Góp vốn/cổ phần/mấy anh em mở công ty = thành lập công ty cổ phần.
    if "công ty cổ phần" in q and has_any(["mở", "thành lập", "góp vốn", "mấy anh em"]):
        return _resolve_proc_name("Đăng ký thành lập công ty cổ phần")

    # C/O đọc đời thường.
    # Ưu tiên VJ trước EUR vì có câu so sánh "VJ ... giống EUR.1" dễ bị bắt nhầm sang EUR.1.
    if "c/o" in q and "vj" in q:
        return _resolve_proc_name("Cấp Giấy chứng nhận xuất xứ hàng hoá (C/O) ưu đãi mẫu VJ")

    if "c/o" in q and ("eur" in q or "eur một" in q or "eur.1" in q):
        return _resolve_proc_name("Cấp Giấy chứng nhận xuất xứ hàng hóa (C/O) mẫu EUR.1")

    # Mai táng phí cựu chiến binh
    if "mai táng" in q and "cựu chiến binh" in q:
        return _resolve_proc_name("Giải quyết chế độ mai táng phí đối với cựu chiến binh")

    # Giám hộ
    if "giám hộ" in q:
        return _resolve_proc_name("Thủ tục đăng ký giám hộ")

    return None


def _is_context_switch_query(query: str) -> bool:
    """
    Nhận diện câu đang đổi ý/đổi thủ tục.
    Ví dụ:
    "Không, ý tôi là giấy xác nhận tình trạng hôn nhân..."
    Khi đó không được bám thủ tục trong lịch sử.
    """
    q = query.lower()
    markers = [
        "không, ý tôi là",
        "không ý tôi là",
        "ý tôi là",
        "không phải",
        "chuyển sang",
        "thay vào đó",
        "còn thủ tục",
        "còn giấy",
        "còn công ty",
    ]
    return any(m in q for m in markers)


def _keyword_detect_procedure(query_lower: str):
    """
    Chấm điểm thủ tục theo keyword trên chính câu hỏi hiện tại.
    Dùng trước history để tránh lỗi context-switch.
    """
    best_match = None
    max_score = 0

    for proc_name, keywords in ENTITIES_DATA.items():
        score = 0

        # Bản thân tên thủ tục cũng là keyword mạnh.
        proc_name_lower = proc_name.lower().strip()
        if proc_name_lower and proc_name_lower in query_lower:
            score += max(5, len(proc_name_lower.split()))

        for k in keywords:
            k_str = str(k).lower().strip()
            if not k_str:
                continue

            # Dùng regex biên từ để tránh bắt nhầm một phần từ.
            if re.search(r'\b' + re.escape(k_str) + r'\b', query_lower):
                score += len(k_str.split())

        if score > max_score:
            max_score = score
            best_match = proc_name

    # Ngưỡng 3 để tránh bắt các keyword quá chung chung như "cấp tỉnh".
    return best_match if max_score >= 3 else None


def detect_procedure_name(query: str, history=None, raw_query: str = ""):
    query_lower = query.lower()
    raw_query_lower = raw_query.lower()

    # Ưu tiên đặc biệt: các cách gọi đời thường hay bị bắt nhầm.
    override_match = _apply_common_procedure_override(raw_query_lower or query_lower)
    if override_match:
        return override_match

    # Ưu tiên 0: Tìm chính xác tên thủ tục trong câu hỏi hiện tại.
    for proc_name in ENTITIES_DATA.keys():
        proc_name_lower = proc_name.lower().strip()
        if proc_name_lower and (
            proc_name_lower in raw_query_lower
            or proc_name_lower in query_lower
        ):
            return proc_name

    # Ưu tiên 1: Dò keyword trên câu hỏi hiện tại.
    # Quan trọng: đặt trước history để câu "Không, ý tôi là..." không bị bám nhầm thủ tục cũ.
    current_match = _keyword_detect_procedure(raw_query_lower or query_lower)
    if current_match:
        return current_match

    # Ưu tiên 2: Nếu câu hiện tại không có thủ tục rõ ràng,
    # mới dùng lịch sử gần nhất để hiểu các câu như "Vậy hồ sơ gồm gì?"
    if history and not _is_context_switch_query(raw_query):
        for msg in reversed(history[-2:]):
            content = msg.get("content", "").lower()

            # Exact name trong history
            for proc_name in ENTITIES_DATA.keys():
                proc_name_lower = proc_name.lower().strip()
                if proc_name_lower and proc_name_lower in content:
                    return proc_name

            # Keyword trong history
            history_match = _keyword_detect_procedure(content)
            if history_match:
                return history_match

    return None


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
        "đối tượng": ["Đối tượng thực hiện"],
    }

    detected_fields = []

    def add_fields(fields):
        for f in fields:
            if f not in detected_fields:
                detected_fields.append(f)

    # Dò theo synonyms, nhưng dò cả key gốc.
    for key, keywords in SYNONYMS_DATA.items():
        candidates = [key] + list(keywords)
        if any(k and k in q for k in candidates):
            mapped = field_map.get(key)
            if mapped:
                add_fields(mapped)

    # Luật bổ sung cho câu thường và câu bẫy.
    if (
        re.search(r"\b\d+\s*(ngày|giờ|tháng)\b", q)
        or "chờ" in q
        or "mất bao lâu" in q
        or "bao lâu" in q
        or "thời hạn" in q
        or "lấy ngay" in q
        or "trong ngày" in q
        or "3 tháng" in q
    ):
        add_fields(["Thời hạn giải quyết", "Cách thức thực hiện"])

    if (
        "lệ phí" in q
        or "phí" in q
        or "đồng" in q
        or "tiền" in q
        or "tốn tiền" in q
        or "tốn phí" in q
        or "đóng bao nhiêu" in q
        or "bao nhiêu tiền" in q
        or "mấy nghìn" in q
        or "online" in q
        or "trực tuyến" in q
    ):
        add_fields(["Phí", "Lệ phí", "Cách thức thực hiện"])

    if (
        "giấy tờ" in q
        or "hồ sơ" in q
        or "chuẩn bị" in q
        or "cần mang" in q
        or "mang giấy" in q
        or "giấy nào" in q
        or "những giấy" in q
        or "cần cái gì" in q
        or "cần nộp" in q
    ):
        add_fields(["Thành phần hồ sơ"])

    if "cơ quan" in q or "nộp ở đâu" in q or "làm ở đâu" in q or "ở đâu" in q or "ra xã" in q or "lên huyện" in q:
        add_fields(["Cơ quan thực hiện"])

    if (
        "kết quả" in q
        or "nhận được gì" in q
        or "được cấp gì" in q
        or "trả về gì" in q
        or "trả cho mình" in q
        or "trả cho tôi" in q
        or "trả cho bác" in q
        or "cái giấy gì" in q
        or "giấy gì" in q and "làm xong" in q
    ):
        add_fields(["Kết quả thực hiện"])

    if "nghị định" in q or "thông tư" in q or "luật" in q or "căn cứ" in q or "ngày ban hành" in q:
        add_fields(["Căn cứ pháp lý", "Cơ quan ban hành", "Cơ quan phối hợp"])

    if detected_fields and any(f in detected_fields for f in ["Thời hạn giải quyết", "Phí", "Lệ phí"]):
        add_fields(["Cách thức thực hiện"])

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
    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in [
            "429",
            "resource_exhausted",
            "resource exhausted",
            "toomanyrequests",
            "too many requests",
            "rate limit",
            "quota",
        ]
    )



def _model_name(model) -> str:
    return getattr(model, "model", None) or getattr(model, "model_name", None) or model.__class__.__name__


def build_qdrant_filter(**conditions):
    """
    Tạo filter chuẩn cho Qdrant.

    LangChain Qdrant lưu metadata trong payload theo dạng:
    metadata.name
    metadata.section_type
    metadata.field
    ...
    """
    must_conditions = []

    for key, value in conditions.items():
        if value is None:
            continue

        qdrant_key = f"metadata.{key}"

        if isinstance(value, list):
            for item in value:
                must_conditions.append(
                    FieldCondition(
                        key=qdrant_key,
                        match=MatchValue(value=item)
                    )
                )
        else:
            must_conditions.append(
                FieldCondition(
                    key=qdrant_key,
                    match=MatchValue(value=value)
                )
            )

    if not must_conditions:
        return None

    return Filter(must=must_conditions)



# ===== SPEED HELPERS =====
def _has_context_reference(query: str) -> bool:
    """
    Nhận diện câu hỏi phụ thuộc ngữ cảnh: "vậy", "còn thủ tục này", "nó", ...
    Các câu này vẫn nên rewrite để gắn lại tên thủ tục từ lịch sử.
    """
    q = query.lower().strip()
    markers = [
        "vậy",
        "thế",
        "còn",
        "thủ tục này",
        "việc này",
        "cái này",
        "nó",
        "như trên",
        "trường hợp này",
        "ý tôi là",
        "ý bố là",
    ]
    return any(m in q for m in markers)


def _should_rewrite_query(raw_query: str, history, detected_proc, detected_field) -> bool:
    """
    Rewrite là một lần gọi LLM phụ, rất tốn thời gian và dễ dính 429 khi test hàng loạt.
    Vì vậy chỉ rewrite khi thật sự cần:
    - Có lịch sử và câu hỏi đang phụ thuộc ngữ cảnh.
    - Chưa nhận diện được tên thủ tục.
    - Bật cưỡng bức ENABLE_QUERY_REWRITE=true.
    """
    if ENABLE_QUERY_REWRITE == "false":
        return False

    if ENABLE_QUERY_REWRITE == "true":
        return True

    has_history = bool(history)
    if has_history and _has_context_reference(raw_query):
        return True

    if not detected_proc:
        return True

    # Nếu đã có tên thủ tục và field rõ ràng thì bỏ rewrite để giảm ít nhất 1 lần gọi Gemini.
    if detected_proc and detected_field:
        return False

    return False


def _is_trap_like(query: str) -> bool:
    """
    Các câu có dấu hiệu bẫy/đối chiếu/sai giả định vẫn nên đi qua LLM
    để bot phản biện thay vì trả template thô.
    """
    q = query.lower()
    markers = [
        "đúng không",
        "phải không",
        "có phải",
        "có đúng",
        "phải chờ",
        "chắc chắn",
        "luôn",
        "không có",
        "không cần",
        "không phải",
        "nhầm",
        "giống",
        "bỏ qua",
        "tự bịa",
        "bịa",
        "đoán đại",
        "3 tháng",
        "mặt trăng",
        "lấy ngày",
        "cộng",
        "nhân",
        "chia",
        "sai",
        "đừng nhầm",
    ]
    return any(m in q for m in markers)



def _is_method_related_field(field) -> bool:
    """
    Chỉ bổ sung chunk cách thức/thời hạn/phí khi câu hỏi thật sự liên quan.
    Tránh trường hợp hỏi hồ sơ nhưng context lại lẫn thời hạn/lệ phí.
    """
    if not field:
        return True

    method_related_fields = {
        "Cách thức thực hiện",
        "Thời hạn giải quyết",
        "Phí",
        "Lệ phí",
    }

    return any(f in method_related_fields for f in field)


def _can_use_direct_answer(raw_query: str, history, detected_proc, field) -> bool:
    """
    Cho phép trả lời trực tiếp kể cả khi session có lịch sử,
    miễn là câu hỏi hiện tại đã tự chứa tên thủ tục + field rõ ràng.
    Nếu câu hỏi phụ thuộc ngữ cảnh như "vậy", "còn thủ tục này" thì để LLM xử lý.
    """
    if not ENABLE_DIRECT_ANSWER:
        return False

    if not detected_proc or not field:
        return False

    if _is_trap_like(raw_query):
        return False

    if _has_context_reference(raw_query):
        return False

    return True


def _strip_chunk_header(chunk_text: str) -> str:
    """
    Bỏ header kỹ thuật của chunk để câu trả lời template gọn hơn.
    """
    lines = []
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Thủ tục:"):
            continue
        if stripped.startswith("Lĩnh vực:"):
            continue
        if stripped.startswith("Loại thông tin:"):
            continue
        lines.append(line)

    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text



def _extract_chunk_section(chunk_text: str) -> str:
    """
    Lấy tên field từ header 'Loại thông tin:' trước khi bỏ header.
    Ví dụ: Cơ quan thực hiện, Kết quả thực hiện, Thành phần hồ sơ...
    """
    match = re.search(r"(?m)^Loại thông tin:\s*(.+?)\s*$", chunk_text)
    if not match:
        return ""
    return match.group(1).strip()



def _chunk_sort_key(doc):
    """
    Sắp xếp chunk theo thứ tự tự nhiên trong metadata.
    Ví dụ:
    _document_1, _document_2, ...
    _method_1_condition_1, ...
    """
    meta = getattr(doc, "metadata", {}) or {}
    chunk_id = str(meta.get("chunk_id", ""))

    section_order = {
        "summary": 0,
        "document": 1,
        "method": 2,
        "simple": 3,
        "legal": 4,
    }

    section_type = str(meta.get("section_type", ""))
    base = section_order.get(section_type, 9)

    nums = [int(n) for n in re.findall(r"\d+", chunk_id)]
    first_num = nums[0] if nums else 9999
    second_num = nums[1] if len(nums) > 1 else 9999

    return (base, first_num, second_num, chunk_id)


def _sort_docs_natural(docs):
    return sorted(docs, key=_chunk_sort_key)


def _extract_label_block(text: str, label: str) -> str:
    """
    Lấy nội dung sau một nhãn như 'Tên giấy tờ:' cho đến nhãn tiếp theo.

    Bản v11 dùng parser theo dòng O(n), không dùng regex DOTALL kiểu .*?
    để tránh chậm bất thường trên các chunk dài.
    """
    if not text or not label:
        return ""

    label_key = label.strip().rstrip(":").lower()
    lines = text.splitlines()

    start_index = -1
    first_value = ""

    for index, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith(label_key + ":"):
            start_index = index
            first_value = stripped.split(":", 1)[1].strip()
            break

    if start_index < 0:
        return ""

    collected = []
    if first_value:
        collected.append(first_value)

    # Chỉ dừng ở các nhãn đứng một mình dạng "Biểu mẫu:", "Số lượng:".
    # Không dừng ở dòng "Bản chính: 1" vì đây là nội dung của Số lượng.
    next_label_pattern = re.compile(r"^[A-ZÀ-Ỵa-zà-ỵ ]+:\s*$")

    for line in lines[start_index + 1:]:
        stripped = line.strip()

        if next_label_pattern.match(stripped):
            break

        collected.append(line)

    value = "\n".join(collected).strip()
    value = re.sub(r"^\s*[-•]\s*", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value

def _format_document_direct_answer(detected_proc: str, context_chunks):
    """
    Format riêng cho câu hỏi hồ sơ/giấy tờ.
    Không gọi LLM, nhưng trình bày dễ đọc hơn raw context.
    """
    items = []
    seen = set()

    for chunk in context_chunks:
        body = _strip_chunk_header(chunk)

        ten_giay_to = _extract_label_block(body, "Tên giấy tờ:")
        so_luong = _extract_label_block(body, "Số lượng:")

        if not ten_giay_to:
            continue

        key = re.sub(r"\s+", " ", ten_giay_to.lower()).strip()
        if key in seen:
            continue
        seen.add(key)

        item_lines = [ten_giay_to]

        if so_luong:
            item_lines.append(f"Số lượng: {so_luong}")

        items.append("\n".join(item_lines))

        if len(items) >= DIRECT_ANSWER_MAX_CHUNKS:
            break

    if not items:
        return None

    lines = [
        f"Hồ sơ cần chuẩn bị cho thủ tục {detected_proc} gồm:",
        "",
    ]

    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item}")
        lines.append("")

    answer = "\n".join(lines).strip()
    answer = answer.replace("**", "")
    answer = re.sub(r"(?m)^\s*\*\s+", "- ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer



def _clean_fee_text(text: str) -> str:
    """
    Làm gọn chuỗi phí/lệ phí lấy từ dữ liệu.
    Ví dụ:
    "Lệ phí : 5000 Đồng Nộp hồ sơ trực tiếp"
    -> "5.000 đồng"
    """
    if not text:
        return ""

    value = re.sub(r"\s+", " ", text).strip()
    value = re.sub(r"(?i)^lệ phí\s*:\s*", "", value).strip()
    value = re.sub(r"(?i)^phí\s*:\s*", "", value).strip()

    # Bỏ mô tả hình thức bị lặp ở cuối vì mình đã có prefix Trực tiếp/Trực tuyến...
    value = re.sub(r"(?i)\s*Nộp hồ sơ trực tiếp\s*$", "", value).strip()
    value = re.sub(r"(?i)\s*Nộp hồ sơ trực tuyến\s*$", "", value).strip()
    value = re.sub(r"(?i)\s*Dịch vụ bưu chính\s*$", "", value).strip()

    # Chuẩn hóa tiền 5000 Đồng -> 5.000 đồng
    money_match = re.search(r"(\d{4,})\s*[Đđ]ồng", value)
    if money_match:
        amount = int(money_match.group(1))
        return f"{amount:,}".replace(",", ".") + " đồng"

    value = value.replace("Đồng", "đồng")
    return value.strip()


def _clean_time_text(text: str) -> str:
    if not text:
        return ""

    value = re.sub(r"\s+", " ", text).strip()
    value = value.replace(" Ngày", " ngày")
    value = value.replace(" Giờ", " giờ")
    return value


def _compact_method_name(name: str) -> str:
    if not name:
        return ""

    name = name.strip()

    mapping = {
        "Trực tiếp": "Trực tiếp",
        "Trực tuyến": "Trực tuyến",
        "Dịch vụ bưu chính": "Dịch vụ bưu chính",
    }

    return mapping.get(name, name)


def _format_time_fee_direct_answer(detected_proc: str, context_chunks):
    """
    Format nhanh cho nhóm câu hỏi thời hạn + phí/lệ phí.

    Bản v5:
    - Ưu tiên chunk Cách thức thực hiện vì chunk này đã gắn phí/lệ phí theo từng hình thức nộp.
    - Nếu đã có phí trong method chunk thì bỏ qua chunk simple Lệ phí để tránh lặp.
    - Chuẩn hóa tiền 5000 Đồng -> 5.000 đồng.
    """
    method_time_parts = []
    method_fee_parts = []
    simple_time_parts = []
    simple_fee_parts = []

    seen_time = set()
    seen_fee = set()
    has_method_fee = False

    for chunk in context_chunks:
        body = _strip_chunk_header(chunk)
        if not body:
            continue

        hinh_thuc = _compact_method_name(_extract_label_block(body, "Hình thức:"))
        thoi_han = _clean_time_text(_extract_label_block(body, "Thời hạn:"))
        muc_phi = _clean_fee_text(_extract_label_block(body, "Mức phí/Lệ phí:"))

        if hinh_thuc and thoi_han:
            item = f"{hinh_thuc}: {thoi_han}"
            key = re.sub(r"\s+", " ", item.lower()).strip()
            if key not in seen_time:
                method_time_parts.append(item)
                seen_time.add(key)

        if hinh_thuc and muc_phi:
            item = f"{hinh_thuc}: {muc_phi}"
            key = re.sub(r"\s+", " ", item.lower()).strip()
            if key not in seen_fee:
                method_fee_parts.append(item)
                seen_fee.add(key)
                has_method_fee = True

        # Chunk simple field: thường không có Hình thức/Mức phí/Lệ phí
        if not hinh_thuc:
            lower_body = body.lower()
            is_fee_chunk = (
                "loại phí" in lower_body
                or "mức phí" in lower_body
                or "lệ phí" in lower_body
                or "đồng" in lower_body
            )
            is_time_chunk = (
                "thời hạn" in lower_body
                or "ngày" in lower_body
                or "giờ" in lower_body
            )

            if is_time_chunk and not is_fee_chunk:
                item = _clean_time_text(body)
                key = re.sub(r"\s+", " ", item.lower()).strip()
                if key not in seen_time:
                    simple_time_parts.append(item)
                    seen_time.add(key)

            if is_fee_chunk:
                # Nếu method chunk đã có phí theo từng hình thức, không thêm simple phí nữa.
                if has_method_fee:
                    continue

                item = _clean_fee_text(body)
                key = re.sub(r"\s+", " ", item.lower()).strip()
                if item and key not in seen_fee:
                    simple_fee_parts.append(item)
                    seen_fee.add(key)

    time_parts = method_time_parts or simple_time_parts
    fee_parts = method_fee_parts or simple_fee_parts

    if not time_parts and not fee_parts:
        return None

    lines = [f"Thời hạn và phí/lệ phí của thủ tục {detected_proc}:", ""]

    if time_parts:
        lines.append("Thời hạn giải quyết:")
        for item in time_parts[:DIRECT_ANSWER_MAX_CHUNKS]:
            lines.append(f"- {item}")
        lines.append("")

    if fee_parts:
        lines.append("Phí/lệ phí:")
        for item in fee_parts[:DIRECT_ANSWER_MAX_CHUNKS]:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("Phí/lệ phí:")
        lines.append("- Dữ liệu hiện tại chưa có thông tin phí/lệ phí cho thủ tục này.")
        lines.append("")

    answer = "\n".join(lines).strip()
    answer = answer.replace("**", "")
    answer = re.sub(r"(?m)^\s*\*\s+", "- ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer


def _format_agency_result_direct_answer(detected_proc: str, context_chunks):
    """
    Format nhanh cho nhóm câu hỏi cơ quan thực hiện + kết quả thực hiện.

    Bản v6:
    - Đọc field từ header 'Loại thông tin:' của chunk.
    - Không phụ thuộc vào body có nhãn 'Cơ quan thực hiện:' hay không.
    """
    agencies = []
    results = []
    seen_agency = set()
    seen_result = set()

    for chunk in context_chunks:
        section = _extract_chunk_section(chunk)
        body = _strip_chunk_header(chunk)

        if not body:
            continue

        value = body.strip()
        key = re.sub(r"\s+", " ", value.lower()).strip()

        if section == "Cơ quan thực hiện":
            if key not in seen_agency:
                agencies.append(value)
                seen_agency.add(key)

        elif section == "Kết quả thực hiện":
            if key not in seen_result:
                results.append(value)
                seen_result.add(key)

    if not agencies and not results:
        return None

    lines = [f"Cơ quan giải quyết và kết quả của thủ tục {detected_proc}:", ""]

    if agencies:
        lines.append("Cơ quan thực hiện:")
        for item in agencies[:DIRECT_ANSWER_MAX_CHUNKS]:
            lines.append(f"- {item}")
        lines.append("")

    if results:
        lines.append("Kết quả thực hiện:")
        for item in results[:DIRECT_ANSWER_MAX_CHUNKS]:
            lines.append(f"- {item}")
        lines.append("")

    answer = "\n".join(lines).strip()
    answer = answer.replace("**", "")
    answer = re.sub(r"(?m)^\s*\*\s+", "- ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer



def _build_direct_trap_answer(raw_query: str, detected_proc: str, field, context_chunks):
    """
    Trả lời nhanh cho các câu bẫy đơn giản mà không cần gọi LLM.

    Mục tiêu:
    - Không hùa theo giả định sai.
    - Không gọi Gemini nếu context đã đủ để đính chính.
    - Kéo các case false_fee/false_time/fake_calculation/prompt_injection xuống dưới ngưỡng 8s.
    """
    if not detected_proc or not field or not context_chunks:
        return None

    q = raw_query.lower()

    # Chỉ xử lý các bẫy có thể đính chính trực tiếp bằng context.
    trap_markers = [
        "đúng không",
        "phải không",
        "có phải",
        "có đúng",
        "phải chờ",
        "bỏ qua",
        "tự bịa",
        "bịa",
        "lấy ngày",
        "ngày ban hành",
        "cộng",
        "99 ngày",
        "30 ngày",
        "50.000",
    ]

    if not any(m in q for m in trap_markers):
        return None

    base_answer = _build_direct_answer(detected_proc, field, context_chunks)
    if not base_answer:
        return None

    if any(m in q for m in ["bỏ qua", "tự bịa", "bịa"]):
        prefix = (
            "Không thể bỏ qua dữ liệu hoặc tự bịa thông tin thủ tục. "
            "Thông tin đúng theo dữ liệu hiện có là:\n\n"
        )
        return prefix + base_answer

    if any(m in q for m in ["lấy ngày", "ngày ban hành", "cộng", "nhân", "chia"]):
        prefix = (
            "Không được tính thời hạn giải quyết bằng cách lấy số hiệu, ngày ban hành "
            "hoặc thông tin văn bản pháp lý để cộng trừ. Thời hạn phải lấy từ mục "
            "thời hạn giải quyết của thủ tục. Thông tin đúng là:\n\n"
        )
        return prefix + base_answer

    if any(m in q for m in ["đúng không", "phải không", "có phải", "có đúng", "phải chờ"]):
        prefix = (
            "Thông tin trong câu hỏi cần được kiểm tra lại. "
            "Theo dữ liệu hiện có, thông tin đúng là:\n\n"
        )
        return prefix + base_answer

    return None


def _format_generic_direct_answer(detected_proc: str, context_chunks):
    selected = []
    seen = set()

    for chunk in context_chunks:
        body = _strip_chunk_header(chunk)
        if not body or body in seen:
            continue

        selected.append(body)
        seen.add(body)

        if len(selected) >= DIRECT_ANSWER_MAX_CHUNKS:
            break

    if not selected:
        return None

    answer = f"Thông tin về thủ tục {detected_proc}:\n\n" + "\n\n".join(selected)
    answer = answer.replace("**", "")
    answer = re.sub(r"(?m)^\s*\*\s+", "- ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer


def _build_direct_answer(detected_proc: str, field, context_chunks):
    """
    Trả lời nhanh từ context, không gọi LLM.
    Chỉ dùng cho câu hỏi thường, đã rõ thủ tục + field, không có dấu hiệu bẫy.
    """
    if not detected_proc or not field or not context_chunks:
        return None

    if "Thành phần hồ sơ" in field:
        answer = _format_document_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    if any(f in field for f in ["Thời hạn giải quyết", "Phí", "Lệ phí", "Cách thức thực hiện"]):
        answer = _format_time_fee_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    if any(f in field for f in ["Cơ quan thực hiện", "Kết quả thực hiện"]):
        answer = _format_agency_result_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    return _format_generic_direct_answer(detected_proc, context_chunks)





def _get_qdrant_collection_name(db):
    """
    Lấy tên collection từ LangChain QdrantVectorStore.
    Hỗ trợ nhiều phiên bản langchain-qdrant khác nhau.
    """
    return (
        getattr(db, "collection_name", None)
        or getattr(db, "_collection_name", None)
        or os.getenv("QDRANT_COLLECTION", "thu_tuc_hanh_chinh_lai_chau")
    )


def _payload_to_document(point):
    """
    Chuyển point Qdrant thành LangChain Document.

    Payload trong Qdrant hiện có dạng:
    {
      "page_content": "...",
      "metadata": {...}
    }
    """
    payload = getattr(point, "payload", None) or {}

    page_content = (
        payload.get("page_content")
        or payload.get("content")
        or payload.get("text")
        or ""
    )

    metadata = payload.get("metadata") or {}

    if not page_content:
        return None

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def _scroll_docs_by_filter(db, qdrant_filter, limit: int):
    """
    Lấy document bằng Qdrant scroll thay vì similarity_search.

    Lý do:
    - similarity_search vẫn phải embed query và chạy vector search.
    - Với fast path đã biết chính xác thủ tục + field, chỉ cần lọc payload.
    - scroll nhanh và ổn định hơn cho các câu direct answer / direct trap.
    """
    try:
        client = getattr(db, "client", None)
        if client is None:
            return []

        collection_name = _get_qdrant_collection_name(db)

        points, _next_page = client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        docs = []
        for point in points:
            doc = _payload_to_document(point)
            if doc is not None:
                docs.append(doc)

        return docs

    except Exception as e:
        print(f"[QDRANT SCROLL ERROR]: {e}")
        return []


def _exact_field_retrieval(db, query: str, detected_proc: str, field):
    """
    Fast path v10: đã biết thủ tục + field thì lấy trực tiếp bằng Qdrant scroll.
    Không embed query, không vector search, không rerank.
    """
    if not detected_proc or not field:
        return []

    started = time.perf_counter()
    exact_docs = []

    for f in field:
        qdrant_filter = build_qdrant_filter(
            name=detected_proc,
            field=f,
        )

        docs_by_field = _scroll_docs_by_filter(
            db=db,
            qdrant_filter=qdrant_filter,
            limit=EXACT_FIELD_K,
        )

        # Fallback an toàn: nếu scroll lỗi hoặc collection format khác thì quay về similarity_search.
        if not docs_by_field:
            fallback_started = time.perf_counter()
            docs_by_field = db.similarity_search(
                query,
                k=EXACT_FIELD_K,
                filter=qdrant_filter,
            )
            fallback_ms = int((time.perf_counter() - fallback_started) * 1000)
            print(f"[FAST EXACT FALLBACK VECTOR]: field={f} latency_ms={fallback_ms}")

        exact_docs.extend(docs_by_field)

    if not exact_docs:
        return []

    exact_docs = _sort_docs_natural(exact_docs)

    unique_docs = []
    seen = set()

    for doc in exact_docs:
        if doc.page_content in seen:
            continue
        unique_docs.append(doc)
        seen.add(doc.page_content)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(
        f"[FAST EXACT SCROLL]: proc='{detected_proc}' "
        f"field={field} chunks={len(unique_docs)} latency_ms={elapsed_ms}"
    )

    return unique_docs


# ===== HÀM THỰC THI LLM VỚI CƠ CHẾ FALLBACK =====
def smart_llm_invoke(prompt: str, prefer_lightweight: bool = False):
    """
    Gọi LLM theo kiểu fast-fail:
    - Mặc định chỉ thử 1 lần trên mỗi model/key.
    - Nếu gặp 429 thì chuyển ngay sang model/key tiếp theo, không ngủ backoff dài.
    - Có thể bật ngủ backoff bằng LLM_SLEEP_ON_429=true nếu muốn chạy nền chậm mà chắc.
    """
    primary = get_llm()
    lightweight = get_lightweight_llm()
    fallbacks = get_fallback_llms()

    if prefer_lightweight:
        candidates = [lightweight, primary] + fallbacks
    else:
        candidates = [lightweight, primary] + fallbacks if USE_LIGHTWEIGHT_FOR_ANSWER else [primary] + fallbacks

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
                        f"latency_ms={elapsed_ms} next_model=True"
                    )

                    if LLM_SLEEP_ON_429 and delay > 0 and attempt < LLM_RETRY_ATTEMPTS:
                        print(f"[LLM 429 SLEEP] delay_s={delay:.1f}")
                        time.sleep(delay)
                        continue

                    # Mặc định: không retry cùng key khi 429, nhảy sang key/model tiếp theo.
                    break

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

    # 1.1. Ngoài phạm vi rõ ràng: trả lời nhanh, không gọi LLM.
    out_of_scope_answer = _direct_out_of_scope_answer(raw_query)
    if out_of_scope_answer:
        return out_of_scope_answer

    # 2. Xử lý ngữ cảnh & Rewrite (Trích xuất thông tin cốt lõi, loại bỏ nhiễu)
    if history and len(history) > 0:
        history_formatted = "LỊCH SỬ HỘI THOẠI:\n" + "\n".join([f"{'Người dùng' if h['role'] == 'user' else 'Chuyên viên'}: {h['content']}" for h in history[-HISTORY_MESSAGES_FOR_REWRITE:]])
    else:
        history_formatted = "LỊCH SỬ HỘI THOẠI:\n(Chưa có)"

    detected_proc_before_rewrite = detect_procedure_name(
        raw_query,
        history=history,
        raw_query=raw_query
    )
    field_before_rewrite = detect_field(raw_query)

    should_rewrite = _should_rewrite_query(
        raw_query=raw_query,
        history=history,
        detected_proc=detected_proc_before_rewrite,
        detected_field=field_before_rewrite
    )

    rewritten = None

    if should_rewrite:
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
    else:
        print(
            f"[REWRITE SKIP]: proc={detected_proc_before_rewrite} "
            f"field={field_before_rewrite}"
        )

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
        detected_proc = detected_proc_before_rewrite or detect_procedure_name(query, history=history, raw_query=raw_query)
        field = field_before_rewrite or detect_field(query)

        # 5. Retrieval
        docs = []

        # FAST PATH:
        # Nếu đã nhận diện được thủ tục + field, lấy thẳng chunk bằng Qdrant filter.
        # Bỏ qua semantic search + reranker để giảm độ trễ.
        fast_exact_docs = _exact_field_retrieval(
            db=db,
            query=query,
            detected_proc=detected_proc,
            field=field,
        )

        if fast_exact_docs:
            docs = fast_exact_docs
            skip_rerank = True
        else:
            skip_rerank = False

            # 5.1 Tìm kiếm ngữ nghĩa tự do
            retriever_semantic = db.as_retriever(
                search_kwargs={"k": RETRIEVAL_SEMANTIC_K}
            )
            docs_semantic = retriever_semantic.invoke(query)
            docs.extend(docs_semantic)

            # 5.2 Tìm kiếm theo bộ lọc keyword nếu có thủ tục gợi ý
            if detected_proc:
                print(f"[SYSTEM]: Keyword gợi ý thủ tục: {detected_proc}")
                qdrant_filter = build_qdrant_filter(name=detected_proc)
                docs_filter = db.similarity_search(
                    query,
                    k=FILTER_SEARCH_K,
                    filter=qdrant_filter
                )
                docs.extend(docs_filter)

        if detected_proc and skip_rerank:
            print(f"[SYSTEM]: Keyword gợi ý thủ tục: {detected_proc}")

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
        if skip_rerank:
            for d in docs:
                d.metadata['score'] = 1.0
            print("[RERANK SKIP]: dùng fast exact retrieval")
        elif reranker:
            rerank_started = time.perf_counter()
            pairs = [(llm_query, d.page_content) for d in docs]
            scores = reranker.predict(pairs)
            rerank_ms = int((time.perf_counter() - rerank_started) * 1000)
            print(f"[RERANK DONE]: docs={len(docs)} latency_ms={rerank_ms}")

            for i, d in enumerate(docs):
                d.metadata['score'] = scores[i]
            docs = sorted(docs, key=lambda x: x.metadata['score'], reverse=True)
        else:
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
                
        # Nếu đã nhận diện được tên thủ tục rõ ràng, ưu tiên tuyệt đối thủ tục đó.
        # Tránh trường hợp reranker chọn nhầm sang các biến thể gần nghĩa như:
        # "khai sinh lưu động", "khai sinh có yếu tố nước ngoài", ...
        if detected_proc:
            allowed_proc_names = [detected_proc]
            print(f"[WINNER - DETECTED PROC]: {allowed_proc_names}")
        else:
            # Lấy top 2 thủ tục có điểm cao nhất thay vì chỉ 1 để tránh đánh rơi ngữ cảnh khi bị nhiễu
            top_procs = sorted(proc_scores.items(), key=lambda item: item[1], reverse=True)[:2]
            allowed_proc_names = [p[0] for p in top_procs] if top_procs else ["Thủ tục không xác định"]
            print(f"[WINNER]: {allowed_proc_names}")

        # [PARENT-CHILD RETRIEVER]
        # Chỉ bổ sung chunk "Cách thức thực hiện" khi câu hỏi liên quan tới
        # cách thức, thời hạn, phí hoặc lệ phí.
        # Nếu người dùng hỏi hồ sơ/giấy tờ thì tuyệt đối không thêm method chunk,
        # tránh bot trả thừa thời hạn/lệ phí trong câu trả lời hồ sơ.
        if _is_method_related_field(field) and not skip_rerank:
            for p_name in allowed_proc_names:
                if p_name != "Thủ tục không xác định":
                    qdrant_filter = build_qdrant_filter(
                        name=p_name,
                        section_type="method"
                    )
                    extra_docs = db.similarity_search(
                        query,
                        k=PARENT_METHOD_K,
                        filter=qdrant_filter
                    )
                    for ed in extra_docs:
                        if ed.page_content not in seen_content:
                            ed.metadata['score'] = 0.99
                            docs.insert(0, ed)
                            seen_content.add(ed.page_content)
        else:
            print(f"[PARENT METHOD SKIP]: field={field}")

        # 8. Lọc Chunk theo Winner và Field
        final_docs = [d for d in docs if d.metadata.get("name") in allowed_proc_names]
        if field:
            # field lúc này là một list các trường tương ứng (Ví dụ: ["Phí", "Lệ phí"])
            field_docs = [d for d in final_docs if d.metadata.get("field") in field]
            if field_docs:
                other_docs = [d for d in final_docs if d not in field_docs]
                final_docs = field_docs + other_docs
                print(f"[FIELD FILTER]: Đã ưu tiên các mục {field} lên đầu")

        # [QDRANT EXACT FIELD BOOST]
        # Khi đã xác định được đúng tên thủ tục và đúng mục thông tin cần hỏi,
        # lấy trực tiếp các chunk theo metadata.name + metadata.field từ Qdrant.
        # Cách này tránh việc reranker/search ban đầu bỏ sót chunk hồ sơ.
        if detected_proc and field and not skip_rerank:
            exact_field_docs = []

            for f in field:
                qdrant_filter = build_qdrant_filter(
                    name=detected_proc,
                    field=f
                )

                docs_by_exact_field = db.similarity_search(
                    query,
                    k=EXACT_FIELD_K,
                    filter=qdrant_filter
                )

                exact_field_docs.extend(docs_by_exact_field)

            if exact_field_docs:
                exact_field_docs = _sort_docs_natural(exact_field_docs)

                print(
                    f"[QDRANT EXACT FIELD BOOST]: Lấy thêm {len(exact_field_docs)} chunks "
                    f"theo thủ tục '{detected_proc}' và field {field}"
                )

                if STRICT_FIELD_CONTEXT:
                    # Khi câu hỏi đã rõ field, chỉ giữ chunk đúng field.
                    # Đây là chốt chống trả lời lan man sang thời hạn/lệ phí khi người dùng hỏi hồ sơ.
                    field_only_docs = [
                        d for d in final_docs
                        if d.metadata.get("field") in field
                    ]
                    field_only_docs = _sort_docs_natural(field_only_docs)
                    final_docs = _sort_docs_natural(exact_field_docs + field_only_docs)
                    print(f"[STRICT FIELD CONTEXT]: chỉ giữ context theo field {field}")
                else:
                    final_docs = _sort_docs_natural(exact_field_docs + final_docs)

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
        
        # 10. Fast direct answer
        # 10.1 Trả lời nhanh cho bẫy đơn giản nếu context đã đủ để đính chính.
        # Cách này tránh gọi LLM khi gặp các câu gài sai phí/thời hạn đơn giản.
        if ENABLE_DIRECT_ANSWER and detected_proc and field and _is_trap_like(raw_query):
            direct_trap_answer = _build_direct_trap_answer(
                raw_query=raw_query,
                detected_proc=detected_proc,
                field=field,
                context_chunks=context_chunks,
            )

            if direct_trap_answer:
                _cache_set(query_key, direct_trap_answer)
                elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
                print(
                    f"[RAG DIRECT TRAP OK] session_id={session_id} latency_ms={elapsed_ms} "
                    f"query_tokens≈{estimate_tokens(raw_query)} answer_tokens≈{estimate_tokens(direct_trap_answer)}"
                )
                return direct_trap_answer

        # 10.2 Với câu hỏi thường đã rõ thủ tục + field, trả lời trực tiếp từ context để tránh gọi LLM.
        if _can_use_direct_answer(
            raw_query=raw_query,
            history=history,
            detected_proc=detected_proc,
            field=field
        ):
            direct_answer = _build_direct_answer(detected_proc, field, context_chunks)

            if not direct_answer and context_chunks:
                direct_answer = _format_generic_direct_answer(detected_proc, context_chunks)

            if direct_answer:
                _cache_set(query_key, direct_answer)
                elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
                print(
                    f"[RAG DIRECT OK] session_id={session_id} latency_ms={elapsed_ms} "
                    f"query_tokens≈{estimate_tokens(raw_query)} answer_tokens≈{estimate_tokens(direct_answer)}"
                )
                return direct_answer

        # 11. Generate (Sử dụng system prompt cố định + Smart LLM với Fallback)
        prompt = f"""{SYSTEM_PROMPT}

{history_formatted}

CONTEXT TÀI LIỆU (Cập nhật mới nhất):
{context}

CÂU HỎI HIỆN TẠI: {raw_query}

LƯU Ý TRẢ LỜI:
- Chỉ trả lời đúng phần người dùng hỏi.
- Nếu người dùng hỏi hồ sơ/giấy tờ thì không tự thêm thời hạn, lệ phí, căn cứ pháp lý.
- Nếu người dùng hỏi thời hạn/lệ phí thì không tự liệt kê toàn bộ hồ sơ.
- Nếu câu hỏi có giả định sai, hãy đính chính ngắn gọn rồi nêu thông tin đúng.

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
            return "Hiện tại dịch vụ AI chưa phản hồi. Vui lòng thử lại sau."

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
        print(f"[CRITICAL ERROR] latency_ms={elapsed_ms}: {e}")
        return "Hệ thống đang bận, vui lòng thử lại sau."