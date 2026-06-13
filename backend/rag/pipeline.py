import time
import json
import re
import os
import random
from collections import OrderedDict
from sentence_transformers import CrossEncoder
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from rag.intent import handle_intent
from rag.normalizer import normalize_query
from rag.config import get_llm, get_lightweight_llm, get_fallback_llms

try:
    from rag.memory import get_selected_procedure, save_selected_procedure
except Exception:
    get_selected_procedure = None
    save_selected_procedure = None

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
DIRECT_ANSWER_MAX_CHUNKS = int(os.getenv("DIRECT_ANSWER_MAX_CHUNKS", "4"))
MULTI_FIELD_MAX_GROUPS = int(os.getenv("MULTI_FIELD_MAX_GROUPS", "4"))
EXACT_PROC_SCROLL_LIMIT = int(os.getenv("EXACT_PROC_SCROLL_LIMIT", "80"))
STRICT_FIELD_CONTEXT = os.getenv("STRICT_FIELD_CONTEXT", "true").lower() == "true"


SYSTEM_PROMPT = """Bạn là trợ lý hướng dẫn thủ tục hành chính cho người dân. Hãy nói chuyện nhẹ nhàng, dễ hiểu, giống một cán bộ một cửa đang chỉ dẫn tận tình.

YÊU CẦU BẮT BUỘC:
1. Chỉ dùng thông tin có trong CONTEXT và lịch sử hội thoại. Không tự bịa thêm quy định, giấy tờ, lệ phí, thời hạn hoặc tên văn bản pháp luật.
2. Trả lời ngắn gọn, đúng ý người hỏi. Ưu tiên 3-5 dòng hoặc vài bước rõ ràng. Chỉ liệt kê dài khi người dùng hỏi đầy đủ hồ sơ hoặc chi tiết.
3. Dùng từ ngữ đời thường. Hạn chế cụm từ nặng như "quy định hiện hành", "căn cứ pháp lý" nếu người dùng không hỏi.
4. Nếu người dùng nói chưa biết làm gì, hãy trấn an trước, rồi hướng dẫn từng bước: chuẩn bị hồ sơ, nơi nộp, thời hạn, phí/lệ phí, nhận kết quả.
5. Nếu hỏi hồ sơ, chỉ trả hồ sơ. Nếu hỏi thời hạn, chỉ trả thời hạn. Nếu hỏi phí/lệ phí, chỉ trả phí/lệ phí. Nếu hỏi nộp ở đâu, chỉ trả nơi nộp/cơ quan xử lý.
6. Nếu câu hỏi có giả định sai, hãy đính chính ngắn gọn và nhẹ nhàng, không làm người dùng thấy bị trách.
7. Nếu dữ liệu chưa có thông tin, nói rõ: "Hiện dữ liệu chưa ghi thông tin này" và gợi ý người dùng hỏi phần khác hoặc liên hệ bộ phận một cửa.
8. Không dùng markdown in đậm như **. Không dùng câu quá dài.
9. Bắt buộc trả lời bằng tiếng Việt.
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


def _clean_proc_id_value(value) -> str:
    """Chuẩn hóa mã thủ tục ở mức kỹ thuật, không thay đổi dữ liệu gốc."""
    return str(value or "").strip()


def _short_procedure_id(value) -> str:
    """
    Lấy mã rút gọn từ mã thủ tục đầy đủ.
    Ví dụ: 2.000575.000.00.00.H35 -> 2.000575.
    """
    proc_id = _clean_proc_id_value(value)
    parts = proc_id.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return proc_id


def _procedure_id_aliases(value):
    """Trả về các biến thể mã thủ tục để so khớp full ID và short ID."""
    proc_id = _clean_proc_id_value(value)
    if not proc_id:
        return []

    aliases = []
    for candidate in [proc_id, _short_procedure_id(proc_id)]:
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return aliases


# Map tên thủ tục -> mã thủ tục để response/frontend có metadata rõ ràng.
# Ưu tiên dùng loader chung để pipeline, admin, frontend cùng nhìn một nguồn dữ liệu.
try:
    try:
        from rag.loader import load_data as _load_procedure_data
        PROCEDURES_DATA = _load_procedure_data()
        print(f"[PIPELINE DATA] Loaded procedures via rag.loader: {len(PROCEDURES_DATA)}")
    except Exception as loader_error:
        print(f"[PIPELINE DATA WARNING - LOADER]: {loader_error}")
        with open("data/procedures.json", "r", encoding="utf-8") as f:
            PROCEDURES_DATA = json.load(f)

    PROCEDURE_ID_BY_NAME = {}
    PROCEDURE_NAME_BY_ID = {}
    for item in PROCEDURES_DATA:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or {}
        proc_name = item.get("name") or content.get("Tên thủ tục") or ""

        # Một số nguồn dữ liệu dùng mã đầy đủ ở item.id nhưng mã rút gọn ở content["Mã thủ tục"].
        # Lưu cả hai để request truyền 2.000575 vẫn khóa được chunk có id 2.000575.000.00.00.H35.
        raw_ids = [
            item.get("id"),
            content.get("Mã thủ tục"),
            item.get("search_code"),
        ]
        proc_id = next((_clean_proc_id_value(x) for x in raw_ids if _clean_proc_id_value(x)), "")

        if proc_name:
            PROCEDURE_ID_BY_NAME[proc_name] = proc_id

        if proc_name:
            for raw_id in raw_ids:
                for alias in _procedure_id_aliases(raw_id):
                    PROCEDURE_NAME_BY_ID[alias] = proc_name
except Exception as e:
    print(f"[FILE LOAD WARNING - PROCEDURES]: {e}")
    PROCEDURES_DATA = []
    PROCEDURE_ID_BY_NAME = {}
    PROCEDURE_NAME_BY_ID = {}

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


def _is_social_intent_query(raw_query: str) -> bool:
    """
    Nhận diện câu xã giao thật sự. Các câu này không cần gợi ý câu hỏi tiếp theo
    và không cần hiển thị metadata thủ tục/nguồn.
    """
    q = (raw_query or "").lower().strip()
    q = re.sub(r"\s+", " ", q)
    social_markers = [
        "xin chào", "chào", "hello", "hi", "alo", "bot ơi",
        "cảm ơn", "thanks", "thank you", "tks",
        "tạm biệt", "bye", "kết thúc", "thoát", "ok xong rồi", "không cần nữa",
    ]
    return any(q == m or q.startswith(m + " ") for m in social_markers)


def _is_acknowledgement_query(raw_query: str) -> bool:
    """
    Người dùng chỉ xác nhận đã hiểu/đã biết. Không kéo thủ tục cũ ra trả lời tiếp.
    Ví dụ: "à", "ừ tôi biết rồi", "tôi biết rồi cảm ơn nhé".
    """
    q = (raw_query or "").lower().strip()
    q = re.sub(r"[.!?…]+$", "", q).strip()
    q = re.sub(r"\s+", " ", q)

    exact = {
        "à", "ừ", "uh", "ừa", "ờ", "ok", "okay", "oki", "uk", "ừm",
        "rồi", "biết rồi", "tôi biết rồi", "tôi hiểu rồi", "hiểu rồi",
        "được rồi", "xong rồi", "thế thôi", "tạm vậy", "ừ tôi biết rồi",
        "ừ tôi hiểu rồi", "tôi biết rồi cảm ơn", "tôi hiểu rồi cảm ơn",
        "tôi biết rồi cảm ơn nhé", "cảm ơn nhé", "cảm ơn nha",
    }
    if q in exact:
        return True

    patterns = [
        r"^(à|ừ|ờ|ok|okay|oki|uk)\s+(tôi|mình|em)?\s*(biết|hiểu)\s+rồi",
        r"^(tôi|mình|em)?\s*(biết|hiểu)\s+rồi(\s+cảm ơn.*)?$",
        r"^(được rồi|xong rồi|ổn rồi)(\s+cảm ơn.*)?$",
    ]
    return any(re.search(p, q) for p in patterns)


def _build_acknowledgement_answer() -> str:
    return (
        "Dạ vâng ạ. Khi cần hỏi tiếp, bạn cứ nhắn ngắn gọn tên giấy tờ "
        "hoặc phần muốn hỏi, mình sẽ hướng dẫn tiếp."
    )


def _is_guidance_query(raw_query: str) -> bool:
    """
    Câu điều hướng hội thoại: người dùng chưa rõ cần làm gì.
    Loại này cần trấn an và dẫn đường, không nên quét Qdrant.
    """
    q = (raw_query or "").lower().strip()
    q = re.sub(r"\s+", " ", q)
    markers = [
        "không biết gì",
        "không biết làm gì",
        "không biết phải làm",
        "chưa biết phải làm",
        "chưa biết làm",
        "tôi chưa biết",
        "tôi không biết",
        "hướng dẫn tôi",
        "hướng dẫn mình",
        "hướng dẫn em",
        "hướng dẫn cho tôi",
        "chỉ tôi",
        "chỉ giúp tôi",
        "chỉ mình",
        "làm đi",
        "bắt đầu từ đâu",
        "tôi phải làm gì",
        "cần làm gì trước",
        "tư vấn từ đầu",
        "nói dễ hiểu",
        "giải thích dễ hiểu",
        "làm như thế nào",
        "làm thế nào",
        "làm như nào",
        "làm sao bây giờ",
        "phải làm thế nào",
        "phải làm như nào",
    ]
    return any(m in q for m in markers)


def _is_low_information_query(raw_query: str) -> bool:
    """
    Câu quá ít thông tin và chưa có thủ tục chính thì hỏi lại,
    không kéo vector DB đi tìm mò.
    """
    q = (raw_query or "").lower().strip()
    if not q:
        return True

    vague_markers = [
        "làm sao",
        "làm thế nào",
        "hướng dẫn",
        "giúp tôi",
        "tư vấn",
        "không biết",
        "cần làm gì",
        "bắt đầu",
        "nói tiếp",
        "rồi sao",
        "tiếp đi",
    ]
    if any(m in q for m in vague_markers):
        return True

    words = [w for w in re.split(r"\s+", q) if w]
    return len(words) <= 3


def _build_guidance_answer(procedure_name: str = "") -> str:
    """
    Trả lời điều hướng thật mềm, giống đang cầm tay chỉ việc.
    Không tự thêm dữ liệu pháp lý ngoài context.
    """
    if not procedure_name:
        return (
            "Không sao ạ, mình cứ đi từng bước.\n\n"
            "Trước hết, bạn cho tôi biết tên giấy tờ hoặc việc cần làm.\n"
            "Ví dụ: đăng ký kết hôn, làm khai sinh, xin giấy xác nhận tình trạng hôn nhân.\n\n"
            "Sau đó tôi sẽ chỉ tiếp: cần giấy tờ gì, nộp ở đâu, bao lâu có kết quả."
        )

    display_name = _clean_procedure_display_name(procedure_name)
    return (
        f"Không sao ạ, tôi sẽ hướng dẫn từ đầu cho thủ tục {display_name}.\n\n"
        "Bạn làm theo thứ tự dễ nhớ như sau:\n"
        "1. Xem mình cần chuẩn bị giấy tờ gì.\n"
        "2. Xem nộp ở cơ quan nào.\n"
        "3. Kiểm tra bao lâu có kết quả.\n"
        "4. Xem có phí/lệ phí hay không.\n"
        "5. Mang hồ sơ đi nộp và chờ nhận kết quả.\n\n"
        "Bạn muốn tôi chỉ trước phần hồ sơ, nơi nộp, hay thời hạn?"
    )


def _popular_procedure_suggestions():
    return [
        "Tôi muốn đăng ký kết hôn",
        "Tôi muốn đăng ký khai sinh",
        "Tôi muốn xin giấy xác nhận tình trạng hôn nhân",
    ]


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



# ===== ROBUST PROCEDURE RESOLVER =====
# Phân biệt ý định hành động của người dân với tên thủ tục trong DB.
# Cơ chế này không fix cứng từng thủ tục, mà áp dụng cho các nhóm dễ nhầm:
# đăng ký/cấp bản sao/cấp lại/sửa đổi/thông báo/điều chỉnh/cấp giấy phép...

_GENERIC_TOKENS = {
    "thủ", "tục", "thu", "tuc", "giấy", "giay", "của", "cho", "về", "ve", "với", "voi",
    "có", "co", "mà", "ma", "thì", "thi",
    "tôi", "toi", "mình", "minh", "em", "anh", "chị", "bác", "ba", "bà", "con", "cái",
    "này", "nay", "kia", "đó", "do", "thế", "the", "vậy", "vay", "nhỉ", "nhi", "ạ", "a",
    "làm", "lam", "xin", "muốn", "muon", "cần", "can", "hỏi", "hoi", "giúp", "giup",
    "bao", "lâu", "lau", "mất", "mat", "phải", "phai", "như", "nhu", "nào", "nao",
    "hồ", "sơ", "so", "nộp", "nop", "ở", "o", "đâu", "dau",
}

_ACTION_KEYWORDS = {
    "register": [
        "đăng ký", "đăng kí", "dang ky", "dang ki", "làm", "lam", "mở", "mo", "thành lập", "thanh lap",
        "lần đầu", "lan dau", "mới sinh", "moi sinh", "cho con", "cho cháu", "cho chau", "em bé", "em be",
    ],
    "copy": [
        "cấp bản sao", "cap ban sao", "xin bản sao", "xin ban sao", "lấy bản sao", "lay ban sao",
        "trích lục", "trich luc", "bản sao", "ban sao", "sao giấy", "sao giay", "sao lại giấy", "sao lai giay",
    ],
    "reissue": [
        "cấp lại", "cap lai", "xin lại", "xin lai", "làm lại", "lam lai", "đổi lại", "doi lai",
        "đăng ký lại", "dang ky lai", "đăng kí lại", "dang ki lai",
        "cấp đổi", "cap doi", "bị mất", "bi mat", "mất giấy", "mat giay", "hỏng", "hong", "rách", "rach",
    ],
    "modify": [
        "sửa đổi", "sua doi", "bổ sung", "bo sung", "điều chỉnh", "dieu chinh", "thay đổi", "thay doi",
        "chuyển mục đích", "chuyen muc dich", "thay thế", "thay the",
    ],
    "notify": ["thông báo", "thong bao", "công bố", "cong bo"],
    "certify": ["xác nhận", "xac nhan", "chứng nhận", "chung nhan", "giấy xác nhận", "giay xac nhan"],
    "license": ["cấp giấy phép", "cap giay phep", "giấy phép", "giay phep", "giấy phép kinh doanh", "đăng ký kinh doanh", "hộ kinh doanh"],
    "support": ["hỗ trợ", "ho tro", "trợ cấp", "tro cap", "mai táng", "mai tang"],
    "terminate": ["thu hồi", "thu hoi", "hủy", "huy", "chấm dứt", "cham dut", "đóng", "dong"],
}

_ACTION_CONFLICTS = {
    "register": {"copy", "reissue", "modify", "notify", "terminate"},
    "copy": {"register", "modify", "notify"},
    "reissue": {"register", "copy", "notify"},
    "modify": {"register", "copy", "notify"},
    "notify": {"register", "modify", "copy", "reissue"},
    "license": {"copy", "notify"},
}


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def _strip_vietnamese_accents(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def _norm_for_match(value: str) -> str:
    value = _strip_vietnamese_accents(value).lower()
    value = re.sub(r"[^a-z0-9/\.\+\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _contains_any_phrase(text: str, phrases) -> bool:
    if not text:
        return False
    raw = _compact_text(text)
    norm = _norm_for_match(text)
    for phrase in phrases:
        if not phrase:
            continue
        p_raw = _compact_text(phrase)
        p_norm = _norm_for_match(phrase)
        if p_raw and p_raw in raw:
            return True
        if p_norm and p_norm in norm:
            return True
    return False


def _meaningful_tokens(text: str):
    norm = _norm_for_match(text)
    tokens = []
    for tok in re.split(r"\s+", norm):
        tok = tok.strip()
        if len(tok) < 2:
            continue
        if tok in _GENERIC_TOKENS:
            continue
        tokens.append(tok)
    return tokens


def _detect_action_families(text: str):
    families = set()
    raw = _compact_text(text)
    norm = _norm_for_match(text)

    for family, phrases in _ACTION_KEYWORDS.items():
        for phrase in phrases:
            p_raw = _compact_text(phrase)
            p_norm = _norm_for_match(phrase)
            if (p_raw and p_raw in raw) or (p_norm and p_norm in norm):
                families.add(family)
                break

    # "mất bao lâu" là hỏi thời hạn, không phải xin cấp lại vì mất giấy.
    if "reissue" in families:
        if any(x in raw for x in ["mất bao lâu", "mat bao lau", "mất mấy ngày", "mat may ngay"]):
            if not _contains_any_phrase(raw, ["cấp lại", "xin lại", "mất giấy", "bị mất", "hỏng", "rách"]):
                families.discard("reissue")

    return families


def _procedure_action_families(procedure_name: str):
    name = procedure_name or ""
    families = _detect_action_families(name)
    raw = _compact_text(name)
    norm = _norm_for_match(name)

    if raw.startswith("đăng ký") or norm.startswith("dang ky"):
        families.add("register")
    if raw.startswith("đăng ký lại") or norm.startswith("dang ky lai"):
        families.add("reissue")
    if raw.startswith("cấp bản sao") or norm.startswith("cap ban sao") or "trích lục" in raw or "trich luc" in norm:
        families.add("copy")
    if raw.startswith("cấp lại") or norm.startswith("cap lai") or raw.startswith("cấp đổi") or norm.startswith("cap doi"):
        families.add("reissue")
    if raw.startswith("điều chỉnh") or norm.startswith("dieu chinh"):
        families.add("modify")
    if raw.startswith("thông báo") or norm.startswith("thong bao"):
        families.add("notify")
    if raw.startswith("cấp giấy") or norm.startswith("cap giay"):
        families.add("license")
    if "xác nhận" in raw or "xac nhan" in norm:
        families.add("certify")

    return families


def _is_negating_family(text: str, family: str) -> bool:
    raw = _compact_text(text)
    norm = _norm_for_match(text)
    negative_starts = ["không phải", "khong phai", "không lấy", "khong lay", "không xin", "khong xin", "sao lại", "sao lai"]

    if family == "copy":
        return any(ns in raw or ns in norm for ns in negative_starts) and _contains_any_phrase(text, ["bản sao", "ban sao", "trích lục", "trich luc"])
    if family == "reissue":
        return any(ns in raw or ns in norm for ns in negative_starts) and _contains_any_phrase(text, ["cấp lại", "cap lai", "xin lại", "xin lai", "làm lại", "lam lai"])

    return False


def _has_explicit_procedure_signal(raw_query: str) -> bool:
    q = raw_query or ""
    if _detect_action_families(q):
        return True
    return _contains_any_phrase(q, [
        "khai sinh", "kết hôn", "giấy độc thân", "tình trạng hôn nhân", "con nuôi", "hộ kinh doanh",
        "khuyến mại", "c/o", "eur", "vj", "rcep", "mai táng", "giám hộ", "công ty", "giấy phép",
    ])



def _phrase_overlap_score(query_text: str, procedure_name: str) -> float:
    """Boost cụm 2-3 từ quan trọng như 'hộ kinh doanh', 'giấy khai sinh', 'mẫu VJ'."""
    q_norm = _norm_for_match(query_text)
    p_tokens = _meaningful_tokens(procedure_name)
    score = 0.0

    # Bigram/trigram trong tên thủ tục, bỏ cụm quá chung.
    phrases = []
    for n in (2, 3):
        for i in range(0, max(0, len(p_tokens) - n + 1)):
            phrase = " ".join(p_tokens[i:i+n])
            if len(phrase) >= 6:
                phrases.append(phrase)

    for phrase in set(phrases):
        if phrase in q_norm:
            score += 18 if len(phrase.split()) == 2 else 26

    return min(score, 60)



_OBJECT_ACTION_TOKENS = {
    "cap", "lai", "dang", "ky", "ki", "xin", "lam", "mo", "thanh", "lap", "bi", "mat", "hong", "rach",
    "doi", "bo", "sung", "sua", "dieu", "chinh", "thay", "the", "thong", "bao", "cong", "bo",
    "giay", "phep", "chung", "nhan", "xac", "nhan",
}


def _object_phrase_alignment_score(query_text: str, procedure_name: str) -> float:
    """
    Giữ đúng đối tượng chính người dân nói.
    Ví dụ query có 'hộ kinh doanh' thì thủ tục thiếu cụm này bị phạt,
    dù có nhiều từ chung như 'cấp lại', 'bị mất', 'kinh doanh'.
    """
    proc_norm = _norm_for_match(procedure_name)
    tokens = [t for t in _meaningful_tokens(query_text) if t not in _OBJECT_ACTION_TOKENS]
    if len(tokens) < 2:
        return 0.0

    phrases2 = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    phrases3 = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)]

    score = 0.0
    if phrases3:
        matched3 = [p for p in phrases3 if p in proc_norm]
        if matched3:
            score += 140 + 25 * min(2, len(matched3))
        else:
            score -= 100

    if phrases2:
        matched2 = [p for p in phrases2 if p in proc_norm]
        if matched2:
            score += 24 * min(3, len(matched2))
        elif not phrases3:
            score -= 25

    return score

def _qualifier_mismatch_penalty(query_text: str, procedure_name: str) -> float:
    """
    Phạt thủ tục có qualifier hẹp mà câu hỏi không nhắc tới.
    Ví dụ: query 'đăng ký khai sinh lần đầu' không nên rơi vào 'đăng ký lại khai sinh có yếu tố nước ngoài'.
    """
    penalty = 0.0
    q_raw = _compact_text(query_text)
    q_norm = _norm_for_match(query_text)
    p_raw = _compact_text(procedure_name)
    p_norm = _norm_for_match(procedure_name)

    def proc_has(phrases):
        return any(_compact_text(x) in p_raw or _norm_for_match(x) in p_norm for x in phrases)

    def query_has(phrases):
        return any(_compact_text(x) in q_raw or _norm_for_match(x) in q_norm for x in phrases)

    qualifier_groups = [
        (["nước ngoài", "yếu tố nước ngoài"], 80),
        (["đăng ký lại", "đăng kí lại"], 55),
        (["người đã có hồ sơ", "hồ sơ, giấy tờ cá nhân", "giấy tờ cá nhân"], 45),
        (["cấp bản sao", "bản sao", "trích lục"], 55),
        (["cấp lại", "cấp đổi"], 45),
        (["sửa đổi", "bổ sung", "điều chỉnh", "thay đổi"], 35),
        (["may rủi"], 25),
        (["thay thế"], 25),
    ]

    for phrases, value in qualifier_groups:
        if proc_has(phrases) and not query_has(phrases):
            penalty += value

    return penalty

def _procedure_match_score(query_text: str, procedure_name: str, keywords=None) -> float:
    q_raw = _compact_text(query_text)
    q_norm = _norm_for_match(query_text)
    name_raw = _compact_text(procedure_name)
    name_norm = _norm_for_match(procedure_name)

    score = 0.0

    if name_raw and name_raw in q_raw:
        score += 120
    elif name_norm and name_norm in q_norm:
        score += 120
    else:
        short_name = re.sub(r"(?i)^thủ tục\s+", "", procedure_name or "").strip()
        short_raw = _compact_text(short_name)
        short_norm = _norm_for_match(short_name)
        if short_raw and short_raw in q_raw:
            score += 95
        elif short_norm and short_norm in q_norm:
            score += 95

    for kw in keywords or []:
        kw_raw = _compact_text(str(kw))
        kw_norm = _norm_for_match(str(kw))
        if not kw_raw:
            continue
        if kw_raw in q_raw or kw_norm in q_norm:
            score += min(40, 6 + len(kw_norm.split()) * 5)

    q_tokens = set(_meaningful_tokens(query_text))
    p_tokens = set(_meaningful_tokens(procedure_name))
    overlap = q_tokens & p_tokens
    score += len(overlap) * 6
    score += _phrase_overlap_score(query_text, procedure_name)
    score += _object_phrase_alignment_score(query_text, procedure_name)
    score -= _qualifier_mismatch_penalty(query_text, procedure_name)

    # Tín hiệu mẫu/mã như EUR.1, VJ, RCEP, LPG, LNG, CNG.
    special_tokens = {t for t in p_tokens if re.search(r"[a-z]+\.?\d*|\d+", t) and len(t) >= 2}
    score += len(special_tokens & q_tokens) * 18

    query_families = _detect_action_families(query_text)
    proc_families = _procedure_action_families(procedure_name)

    for fam in ["copy", "reissue"]:
        if fam in proc_families and _is_negating_family(query_text, fam):
            score -= 90

    if query_families:
        matched = query_families & proc_families
        if matched:
            score += 35 + len(matched) * 8

        conflicts = set()
        for fam in query_families:
            conflicts |= _ACTION_CONFLICTS.get(fam, set())
        if proc_families & conflicts:
            if not matched:
                score -= 55
            else:
                score -= 18

    if _contains_any_phrase(query_text, ["lần đầu", "lan dau", "mới làm", "moi lam"]):
        if proc_families & {"copy", "reissue", "modify"}:
            score -= 80
        if "register" in proc_families:
            score += 35

    return score



_PROCEDURE_INDEX_CACHE = None


def _get_procedure_index():
    global _PROCEDURE_INDEX_CACHE
    if _PROCEDURE_INDEX_CACHE is not None:
        return _PROCEDURE_INDEX_CACHE

    names = list(ENTITIES_DATA.keys())
    for name in PROCEDURE_ID_BY_NAME.keys():
        if name not in ENTITIES_DATA:
            names.append(name)

    index = []
    for name in names:
        short_name = re.sub(r"(?i)^thủ tục\s+", "", name or "").strip()
        p_tokens = set(_meaningful_tokens(name))
        special_tokens = {t for t in p_tokens if re.search(r"[a-z]+\.?\d*|\d+", t) and len(t) >= 2}
        index.append({
            "name": name,
            "raw": _compact_text(name),
            "norm": _norm_for_match(name),
            "short_raw": _compact_text(short_name),
            "short_norm": _norm_for_match(short_name),
            "tokens": p_tokens,
            "special_tokens": special_tokens,
            "families": _procedure_action_families(name),
        })

    _PROCEDURE_INDEX_CACHE = index
    return index


def _procedure_match_score_index(query_text: str, item: dict) -> float:
    q_raw = _compact_text(query_text)
    q_norm = _norm_for_match(query_text)
    q_tokens = set(_meaningful_tokens(query_text))

    score = 0.0
    if item["raw"] and item["raw"] in q_raw:
        score += 120
    elif item["norm"] and item["norm"] in q_norm:
        score += 120
    elif item["short_raw"] and item["short_raw"] in q_raw:
        score += 95
    elif item["short_norm"] and item["short_norm"] in q_norm:
        score += 95

    overlap = q_tokens & item["tokens"]
    score += len(overlap) * 6
    score += _phrase_overlap_score(query_text, item["name"])
    score += _object_phrase_alignment_score(query_text, item["name"])
    score -= _qualifier_mismatch_penalty(query_text, item["name"])
    score += len(item["special_tokens"] & q_tokens) * 18

    query_families = _detect_action_families(query_text)
    proc_families = item["families"]

    for fam in ["copy", "reissue"]:
        if fam in proc_families and _is_negating_family(query_text, fam):
            score -= 90

    if query_families:
        matched = query_families & proc_families
        if matched:
            score += 35 + len(matched) * 8

        conflicts = set()
        for fam in query_families:
            conflicts |= _ACTION_CONFLICTS.get(fam, set())
        if proc_families & conflicts:
            if not matched:
                score -= 55
            else:
                score -= 18

    if _contains_any_phrase(query_text, ["lần đầu", "lan dau", "mới làm", "moi lam"]):
        if proc_families & {"copy", "reissue", "modify"}:
            score -= 80
        if "register" in proc_families:
            score += 35

    return score

def _robust_detect_procedure_from_current_query(raw_query: str):
    if not raw_query:
        return None

    candidates = []
    for item in _get_procedure_index():
        score = _procedure_match_score_index(raw_query, item)
        if score > 0:
            candidates.append((item["name"], score))

    if not candidates:
        return None

    # Nếu điểm bằng/sát nhau, ưu tiên tên thủ tục ngắn/gốc hơn để tránh rơi vào biến thể hẹp.
    candidates.sort(key=lambda item: (item[1], -len(item[0])), reverse=True)
    best_name, best_score = candidates[0]
    second_score = candidates[1][1] if len(candidates) > 1 else 0

    if best_score < 18:
        return None

    if second_score and (best_score - second_score) < 6 and not _detect_action_families(raw_query):
        print(f"[PROC AMBIGUOUS]: top1={best_name}:{best_score:.1f} top2={candidates[1][0]}:{second_score:.1f}")
        return None

    print(f"[PROC RESOLVER]: selected={best_name} score={best_score:.1f} second={second_score:.1f}")
    return _resolve_proc_name(best_name) or best_name

def _is_birth_registration_intent(q: str) -> bool:
    """
    Nhận diện câu muốn đăng ký khai sinh lần đầu.
    Đây là nhóm người dân hay nói không chuẩn:
    - đăng ký giấy khai sinh
    - làm giấy khai sinh
    - khai sinh lần đầu
    - làm khai sinh cho con/cháu

    Quan trọng: nếu câu có chữ "bản sao" nhưng đang phủ định/sửa lỗi
    kiểu "sao lại bản sao, tôi muốn đăng ký giấy khai sinh" thì vẫn phải
    quay về thủ tục đăng ký khai sinh, không bám thủ tục cấp bản sao.
    """
    q = (q or "").lower()

    correction_markers = [
        "sao lại bản sao",
        "không phải bản sao",
        "không lấy bản sao",
        "không xin bản sao",
        "tôi đang muốn đăng ký",
        "ý tôi là đăng ký",
        "đăng ký lần đầu",
        "lần đầu",
        "khai sinh lần đầu",
    ]

    registration_markers = [
        "đăng ký giấy khai sinh",
        "đăng kí giấy khai sinh",
        "đăng ký khai sinh",
        "đăng kí khai sinh",
        "làm giấy khai sinh",
        "làm khai sinh",
        "khai sinh cho",
        "khai sinh lần đầu",
        "giấy khai sinh lần đầu",
        "cho con",
        "cho cháu",
        "trẻ con",
        "cháu bé",
        "em bé",
        "mới sinh",
    ]

    if "khai sinh" not in q:
        return False

    if any(m in q for m in correction_markers):
        return True

    return any(m in q for m in registration_markers)


def _is_birth_copy_intent(q: str) -> bool:
    """
    Nhận diện câu thật sự muốn cấp bản sao/trích lục giấy khai sinh.
    Không dùng mỗi cụm "giấy khai sinh" để chọn bản sao, vì người dân hay nói
    "đăng ký giấy khai sinh" để chỉ đăng ký khai sinh lần đầu.
    """
    q = (q or "").lower()

    if "khai sinh" not in q:
        return False

    # Nếu trong câu có dấu hiệu đăng ký lần đầu/sửa lỗi thì tuyệt đối không chọn bản sao.
    if _is_birth_registration_intent(q):
        return False

    copy_markers = [
        "cấp bản sao",
        "xin bản sao",
        "lấy bản sao",
        "bản sao giấy khai sinh",
        "sao giấy khai sinh",
        "trích lục khai sinh",
        "trích lục hộ tịch",
        "cấp lại giấy khai sinh",
        "xin lại giấy khai sinh",
        "mất giấy khai sinh",
        "hỏng giấy khai sinh",
        "rách giấy khai sinh",
    ]

    return any(m in q for m in copy_markers)


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

    # Khai sinh là nhóm dễ nhầm nhất:
    # - "đăng ký giấy khai sinh", "khai sinh lần đầu" => đăng ký khai sinh
    # - "cấp bản sao", "trích lục", "xin lại" => cấp bản sao/trích lục
    if "khai sinh" in q:
        if _is_birth_registration_intent(q):
            matched = _resolve_proc_name("Thủ tục đăng ký khai sinh")
            if matched:
                return matched

        if _is_birth_copy_intent(q):
            matched = _resolve_proc_name("Cấp bản sao Trích lục hộ tịch, bản sao Giấy khai sinh")
            if matched:
                return matched

        # Nếu người dân nói chung chung "giấy khai sinh" nhưng có động từ hỏi làm/nộp/mất bao lâu,
        # ưu tiên đăng ký khai sinh vì đây là nhu cầu phổ biến và an toàn hơn bản sao.
        if has_any(["mất bao lâu", "bao lâu", "làm sao", "làm thế nào", "phải làm", "nộp ở đâu", "cần gì", "hồ sơ"]):
            matched = _resolve_proc_name("Thủ tục đăng ký khai sinh")
            if matched:
                return matched

    # Cưới vợ/chồng/giấy cưới = đăng ký kết hôn, trừ khi câu đang hỏi giấy độc thân đã bắt ở trên.
    # Đây là nhóm alias đời thường, chỉ dùng để chọn đúng thủ tục, không dùng để tự bịa câu trả lời.
    if has_any([
        "cưới vợ", "cưới chồng", "đăng ký kết hôn", "đăng kí kết hôn",
        "kết hôn", "giấy cưới", "làm giấy cưới", "đăng ký cưới", "đăng kí cưới",
        "giấy kết hôn", "chứng nhận kết hôn",
    ]):
        return _resolve_proc_name("Thủ tục đăng ký kết hôn")

    # Câu hỏi quan hệ họ hàng đi kèm ý định kết hôn/cưới cũng phải neo về đăng ký kết hôn,
    # tránh resolver tổng quát bắt nhầm sang thủ tục đất đai/doanh nghiệp có nhiều token chung.
    if has_any(["con chú", "con bác", "anh em họ", "họ hàng", "ba đời", "phạm vi ba đời"]) and has_any(["kết hôn", "cưới", "đăng ký", "đăng kí"]):
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
    if "công ty cổ phần" in q and has_any(["m mở", "thành lập", "góp vốn", "mấy anh em"]):
        return _resolve_proc_name("Đăng ký thành lập công ty cổ phần")

    # C/O đọc đời thường.
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
        "ý là",
        "không phải",
        "sao lại",
        "cơ mà",
        "tôi đang muốn",
        "mình đang muốn",
        "em đang muốn",
        "quay lại",
        "trở lại",
        "lần đầu",
        "chuyển sang",
        "thay vào đó",
        "còn thủ tục",
        "còn giấy",
        "còn công ty",
    ]
    return any(m in q for m in markers)


def _has_hard_context_anchor(raw_query: str) -> bool:
    """
    Dấu hiệu người dùng đang hỏi nối tiếp thủ tục đã chọn.
    Khác với _has_context_reference(), hàm này không coi riêng chữ "còn"
    là đủ mạnh, vì "còn thủ tục hộ kinh doanh..." có thể là chuyển chủ đề.
    """
    q = (raw_query or "").lower().strip()
    q = re.sub(r"\s+", " ", q)

    hard_markers = [
        "thủ tục này",
        "việc này",
        "cái này",
        "cái đó",
        "trường hợp này",
        "như trên",
        "như vậy",
        "vậy thì",
        "thế thì",
        "thế còn",
        "nó",
        "giấy này",
        "hai loại giấy này",
        "mấy giấy này",
        "phần này",
        "còn cái này",
        "lệ phí",
        "phí",
        "thời hạn",
        "bao lâu",
        "nộp ở đâu",
        "ra xã",
        "bưu điện",
        "online",
        "hồ sơ",
        "giấy tờ",
    ]
    return any(marker in q for marker in hard_markers)


def _has_strong_switch_phrase(raw_query: str) -> bool:
    """
    Chỉ các cụm thật sự báo hiệu đổi thủ tục mới được xem là switch mạnh.
    Không dùng các cụm mơ hồ như "không phải" hoặc "lần đầu" vì chúng hay xuất hiện
    trong câu bẫy/đính chính của cùng một thủ tục.
    """
    q = (raw_query or "").lower().strip()
    q = re.sub(r"\s+", " ", q)

    markers = [
        "chuyển sang",
        "thay vào đó",
        "sang thủ tục",
        "hỏi thủ tục khác",
        "thủ tục khác",
        "còn thủ tục",
        "không phải thủ tục này",
        "không hỏi thủ tục này",
        "tôi muốn hỏi thủ tục",
        "mình muốn hỏi thủ tục",
        "em muốn hỏi thủ tục",
    ]
    return any(marker in q for marker in markers)


def _explicit_proc_match_strength(raw_query: str, explicit_proc: str) -> float:
    """
    Ước lượng độ chắc của thủ tục vừa resolver ra từ chính câu hiện tại.
    Điểm thấp nghĩa là resolver đang đoán theo vài token nhiễu, không nên thắng
    selected_procedure trong phiên.
    """
    if not raw_query or not explicit_proc:
        return 0.0

    try:
        return float(_procedure_match_score(raw_query, explicit_proc))
    except Exception:
        return 0.0


def _should_keep_selected_context(raw_query: str, selected_proc_name: str = "", explicit_proc: str = "", field=None) -> bool:
    """
    Luật chốt ngữ cảnh hội thoại.

    Khi phiên chat đã có selected_procedure, các câu hỏi nối tiếp có field rõ
    hoặc có đại từ ngữ cảnh phải bám selected_procedure. Resolver chỉ được đổi
    thủ tục nếu câu hiện tại có tín hiệu chuyển thủ tục đủ mạnh.
    """
    if not selected_proc_name:
        return False

    if explicit_proc and _same_procedure_name(explicit_proc, selected_proc_name):
        return True

    if _has_hard_context_anchor(raw_query):
        return True

    # Nếu câu có field rõ như hồ sơ/phí/thời hạn/nơi nộp nhưng không hề có
    # cụm chuyển thủ tục mạnh, coi là hỏi tiếp thủ tục đang chọn.
    if field and not _has_strong_switch_phrase(raw_query):
        strength = _explicit_proc_match_strength(raw_query, explicit_proc) if explicit_proc else 0.0

        # Nếu resolver chỉ đoán yếu, không cho nó cướp ngữ cảnh.
        if not explicit_proc or strength < 85:
            return True

        # Nếu câu không có bằng chứng nêu thủ tục mới đủ rõ, vẫn giữ ngữ cảnh cũ.
        if not _has_explicit_procedure_signal(raw_query):
            return True

    # Câu điều hướng/cụt như "hướng dẫn tiếp", "làm sao bây giờ" luôn bám session.
    if _is_guidance_query(raw_query) and not _has_strong_switch_phrase(raw_query):
        return True

    return False


def _keyword_detect_procedure(query_lower: str):
    """
    Fallback keyword scoring cũ, nhưng có thêm điểm/phạt theo action family.
    Hàm chính vẫn là _robust_detect_procedure_from_current_query().
    """
    best_match = None
    max_score = 0

    for proc_name, keywords in ENTITIES_DATA.items():
        score = _procedure_match_score(query_lower, proc_name, keywords)
        if score > max_score:
            max_score = score
            best_match = proc_name

    return best_match if max_score >= 18 else None


def detect_procedure_name(query: str, history=None, raw_query: str = ""):
    query_lower = (query or "").lower()
    raw_query_lower = (raw_query or query or "").lower()
    current_text = raw_query or query or ""

    # Ưu tiên 0: một số cách gọi đời thường cực phổ biến, độ tin cậy cao.
    # Đặt trước robust resolver để tránh các câu ngắn/đời thường bị token chung kéo sang thủ tục khác.
    override_match = _apply_common_procedure_override(raw_query_lower or query_lower)
    if override_match:
        return override_match

    # Ưu tiên 1: resolver tổng quát chỉ dựa vào câu hiện tại.
    robust_match = _robust_detect_procedure_from_current_query(current_text)
    if robust_match:
        return robust_match

    # Ưu tiên 2: tìm chính xác tên thủ tục trong câu hỏi hiện tại.
    for proc_name in ENTITIES_DATA.keys():
        proc_name_lower = proc_name.lower().strip()
        if proc_name_lower and (
            proc_name_lower in raw_query_lower
            or proc_name_lower in query_lower
        ):
            return proc_name

    # Ưu tiên 3: dò keyword/action scoring trên câu hỏi hiện tại.
    current_match = _keyword_detect_procedure(raw_query_lower or query_lower)
    if current_match:
        return current_match

    # Ưu tiên 4: chỉ dùng lịch sử khi câu hiện tại không phải câu sửa lỗi/đổi thủ tục.
    if history and not _is_context_switch_query(raw_query or query):
        for msg in reversed(history[-2:]):
            content = msg.get("content", "")
            history_match = _robust_detect_procedure_from_current_query(content)
            if history_match:
                return history_match

            content_lower = content.lower()
            for proc_name in ENTITIES_DATA.keys():
                proc_name_lower = proc_name.lower().strip()
                if proc_name_lower and proc_name_lower in content_lower:
                    return proc_name

            history_keyword_match = _keyword_detect_procedure(content_lower)
            if history_keyword_match:
                return history_keyword_match

    return None


def _strip_procedure_mention_for_intent(query: str, procedure_name: str = "") -> str:
    """
    Bỏ tên thủ tục khỏi câu hỏi trước khi dò field/ý định.

    Nhiều tên thủ tục có chứa từ khóa field, ví dụ "mai táng phí".
    Nếu quét trực tiếp toàn câu, mọi câu hỏi về thủ tục này sẽ bị hiểu nhầm
    là hỏi phí/lệ phí.
    """
    value = (query or "").lower()
    if not value:
        return ""

    variants = set()
    if procedure_name:
        proc = (procedure_name or "").strip()
        if proc:
            variants.add(proc)
            without_prefix = re.sub(r"(?i)^thủ tục\s+", "", proc).strip()
            variants.add(without_prefix)
            variants.add("thủ tục " + without_prefix)
            try:
                variants.add(_clean_procedure_display_name(proc))
            except Exception:
                pass

    for name in sorted([v for v in variants if v], key=len, reverse=True):
        value = re.sub(re.escape(name.lower()), " ", value, flags=re.IGNORECASE)

    value = re.sub(r"\bthủ\s+tục\b\s*:?", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def detect_field(query: str, procedure_name: str = ""):
    q = _strip_procedure_mention_for_intent(query, procedure_name)

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

    if any(x in q for x in [
        "hạn sử dụng",
        "còn hạn",
        "dưới 6 tháng",
        "trong 6 tháng",
        "giấy khám sức khỏe",
        "giấy xác nhận thu nhập",
    ]):
        add_fields(["Thành phần hồ sơ", "Yêu cầu điều kiện"])

    # Luật bổ sung cho câu thường và câu bẫy.
    # Điều kiện thực hiện phải được nhận diện trước thời hạn.
    # Ví dụ "19 tuổi 11 tháng" là hỏi điều kiện tuổi, không phải thời hạn xử lý hồ sơ.
    condition_markers = [
        "điều kiện", "đủ điều kiện", "được chưa", "được không", "có được",
        "đủ tuổi", "tuổi", "tự nguyện", "cấm", "bị cấm",
        "anh em", "con chú", "con bác", "anh em họ", "họ hàng", "ba đời", "phạm vi ba đời",
    ]
    is_condition_like = any(m in q for m in condition_markers)
    is_age_or_kinship_context = any(m in q for m in [
        "tuổi", "đủ tuổi", "tự nguyện", "anh em", "con chú", "con bác", "anh em họ", "họ hàng", "ba đời", "phạm vi ba đời",
    ])
    if is_condition_like:
        add_fields(["Yêu cầu điều kiện"])

    # Tránh nhầm "tháng" trong "hạn sử dụng dưới 6 tháng" thành "Thời hạn giải quyết".
    if (
        (not is_age_or_kinship_context and re.search(r"\b\d+\s*(ngày|giờ)\b", q))
        or (re.search(r"\b\d+\s*(tháng)\b", q) and not any(x in q for x in ["hạn", "sử dụng", "hạn sử dụng"]))
        or "chờ" in q
        or "mất bao lâu" in q
        or "bao lâu" in q
        or "thời hạn" in q and not any(x in q for x in ["hạn sử dụng", "hạn giấy", "còn hạn"])
        or "lấy ngay" in q
        or "trong ngày" in q
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
    ):
        add_fields(["Phí", "Lệ phí", "Cách thức thực hiện"])

    if (
        "online" in q
        or "trực tuyến" in q
        or "nộp qua mạng" in q
        or "nộp trực tuyến" in q
        or "dịch vụ công" in q
        or "bưu chính" in q
        or "qua bưu điện" in q
        or "hình thức" in q
        or "cách nộp" in q
        or "cách thức" in q
    ):
        add_fields(["Cách thức thực hiện"])

    if (
        "giấy tờ" in q
        or "hồ sơ" in q
        or "chuẩn bị" in q
        or "cần mang" in q
        or "mang giấy" in q
        or "giấy nào" in q
        or "những giấy" in q
        or "cần cái gì" in q
        or (
            "cần nộp" in q
            and not any(x in q for x in [
                "ở đâu",
                "nộp ở đâu",
                "nộp tại đâu",
                "nộp chỗ nào",
                "cơ quan nào",
                "làm ở đâu",
            ])
        )
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



# ===== CONVERSATION-AWARE QUERY ROUTER =====
# Router này dùng "bằng chứng dương tính" thay vì liệt kê cứng các chủ đề ngoài phạm vi.
# Mục tiêu:
# - Không cắt ngang câu hỏi nối tiếp của người dân.
# - Không để ngữ cảnh thủ tục cũ kéo câu hỏi lệch chủ đề vào RAG.
# - Không gọi Qdrant/LLM khi chưa có đủ dấu hiệu thuộc miền thủ tục hành chính.

ROUTE_ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
ROUTE_SOCIAL = "SOCIAL"
ROUTE_CONTINUE_CONTEXT = "CONTINUE_CONTEXT"
ROUTE_SWITCH_PROCEDURE = "SWITCH_PROCEDURE"
ROUTE_ADMIN_CONFIDENT = "ADMIN_CONFIDENT"
ROUTE_ASK_CLARIFY = "ASK_CLARIFY"
ROUTE_OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
ROUTE_LIST_QUERY = "LIST_QUERY"

_ADMIN_DOMAIN_PHRASES = [
    # Dấu hiệu chung của thủ tục hành chính
    "thủ tục", "tthc", "hồ sơ", "giấy tờ", "mẫu đơn", "biểu mẫu",
    "bản chính", "bản sao", "trích lục", "nộp hồ sơ", "nộp ở đâu",
    "cơ quan", "ủy ban", "ubnd", "bộ phận một cửa", "dịch vụ công",
    "thời hạn", "bao lâu có kết quả", "lệ phí", "phí", "kết quả thực hiện",
    "căn cứ pháp lý", "nghị định", "thông tư", "quyết định",

    # Hành động hành chính đủ đặc trưng
    "đăng ký", "đăng kí", "cấp lại", "cấp đổi", "cấp bản sao",
    "xin bản sao", "xin giấy", "cấp giấy", "giấy phép", "giấy chứng nhận",
    "xác nhận", "chứng nhận", "thông báo", "công bố", "điều chỉnh",
    "sửa đổi", "bổ sung", "thành lập", "giải quyết chế độ",
]


def _query_has_admin_positive_evidence(raw_query: str, field=None, explicit_proc: str = "") -> bool:
    """
    Không hỏi: "câu này có nằm trong danh sách off-topic không?"
    Mà hỏi: "câu này có đủ bằng chứng là đang hỏi thủ tục hành chính không?"

    Nếu chỉ có bằng chứng yếu mà chưa có selected_procedure, hệ thống sẽ hỏi lại,
    không tìm mò toàn bộ Qdrant.
    """
    q = (raw_query or "").lower().strip()
    q = re.sub(r"\s+", " ", q)

    if explicit_proc:
        return True

    if field:
        return True

    if _contains_any_phrase(q, _ADMIN_DOMAIN_PHRASES):
        return True

    # Một số tên lĩnh vực/đối tượng hay xuất hiện trong câu hỏi người dân,
    # nhưng vẫn chỉ là bằng chứng để hỏi lại nếu chưa xác định được thủ tục.
    domain_subjects = [
        "khai sinh", "kết hôn", "giấy độc thân", "tình trạng hôn nhân",
        "con nuôi", "giám hộ", "hộ kinh doanh", "công ty", "doanh nghiệp",
        "khuyến mại", "đất đai", "nhà ở", "mai táng", "c/o", "hộ tịch",
    ]
    return _contains_any_phrase(q, domain_subjects)


def _is_valid_context_followup(raw_query: str, field=None) -> bool:
    """
    Câu nối tiếp hợp lệ khi đã có selected_procedure.
    Đây là lớp bảo vệ để không cắt lời người dân khi họ hỏi cụt:
    "thế phí thì sao", "nộp ở đâu", "tôi chưa biết làm thế nào".
    """
    q = (raw_query or "").lower().strip()
    q = re.sub(r"\s+", " ", q)

    if not q:
        return False

    if field:
        return True

    if _is_guidance_query(q):
        return True

    if _has_context_reference(q):
        return True

    followup_markers = [
        "nói tiếp", "tiếp đi", "rồi sao", "còn nữa không", "làm tiếp",
        "chỉ tiếp", "hướng dẫn tiếp", "phần tiếp theo", "cho tôi biết tiếp",
        "vậy làm sao", "vậy làm thế nào", "thế làm sao", "thế làm thế nào",
        "cái đó", "cái này", "thủ tục này", "việc này",
    ]
    if _contains_any_phrase(q, followup_markers):
        return True

    # Câu rất ngắn nhưng có từ nối tiếp thường là đang bám cuộc trò chuyện.
    words = [w for w in re.split(r"\s+", q) if w]
    if len(words) <= 5 and any(x in q for x in ["thế", "vậy", "còn", "tiếp"]):
        return True

    return False




def _is_related_list_query(raw_query: str) -> bool:
    """
    Nhận diện câu hỏi yêu cầu danh sách/đếm nhóm thủ tục.

    Điểm quan trọng: đây không phải câu hỏi về một thủ tục riêng lẻ, nên không được
    để selected_context hoặc resolver kéo về một thủ tục cụ thể.
    Ví dụ:
    - còn thủ tục nào khác liên quan đến đất đai không, liệt kê cho tôi xem
    - có bao nhiêu thủ tục hộ tịch
    - cho tôi danh sách các thủ tục về khuyến mại
    """
    q = _compact_text(raw_query)
    if not q:
        return False

    list_markers = [
        "danh sách", "danh sach", "liệt kê", "liet ke", "kể ra", "ke ra",
        "các thủ tục", "cac thu tuc", "những thủ tục", "nhung thu tuc",
        "thủ tục nào", "thu tuc nao", "bao nhiêu thủ tục", "bao nhieu thu tuc",
        "còn thủ tục", "con thu tuc", "thủ tục khác", "thu tuc khac",
    ]
    relation_markers = [
        "liên quan", "lien quan", "thuộc lĩnh vực", "thuoc linh vuc",
        "lĩnh vực", "linh vuc", "về", "ve", "trong nhóm", "trong nhom",
    ]

    has_list_marker = _contains_any_phrase(q, list_markers)
    has_relation_marker = _contains_any_phrase(q, relation_markers)

    # Có marker danh sách rõ ràng là đủ.
    if has_list_marker:
        return True

    # Câu kiểu "đất đai có những gì" cũng nên coi là hỏi danh sách nếu có từ nhiều/các/những.
    if has_relation_marker and any(x in q for x in ["những", "nhung", "các", "cac", "nào", "nao"]):
        return True

    return False



def _clean_list_topic_text(text: str) -> str:
    """
    Làm sạch phần chủ đề trong câu hỏi dạng danh sách.
    Ví dụ: "cho tôi danh sách các thủ tục kết hôn đi" -> "kết hôn".
    """
    value = _compact_text(text)
    if not value:
        return ""

    remove_phrases = [
        "cho tôi", "cho toi", "cho mình", "cho minh", "cho em", "cho anh", "cho chị", "cho chi", "cho bác", "cho bac",
        "xem", "xem đi", "di", "đi", "nhé", "nhe", "ạ", "a", "được không", "duoc khong",
        "danh sách", "danh sach", "liệt kê", "liet ke", "kể ra", "ke ra",
        "các thủ tục", "cac thu tuc", "những thủ tục", "nhung thu tuc", "mấy thủ tục", "may thu tuc",
        "thủ tục nào", "thu tuc nao", "thủ tục khác", "thu tuc khac", "còn thủ tục nào khác", "con thu tuc nao khac",
        "còn thủ tục", "con thu tuc", "bao nhiêu thủ tục", "bao nhieu thu tuc",
        "liên quan đến", "lien quan den", "liên quan tới", "lien quan toi", "có liên quan đến", "co lien quan den",
        "thuộc lĩnh vực", "thuoc linh vuc", "lĩnh vực", "linh vuc", "trong nhóm", "trong nhom",
        "về", "ve", "với", "voi", "không", "khong", "có", "co",
    ]

    # Xóa cụm dài trước, tránh để sót vụn từ.
    # Dùng ranh giới từ để không xóa nhầm bên trong từ khác:
    # ví dụ không được xóa "đi" trong "đất đai", hoặc "a" trong "mại".
    for phrase in sorted(remove_phrases, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        value = re.sub(pattern, " ", value) 

    value = re.sub(r"[,.?;:!]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _extract_related_list_topic(raw_query: str) -> str:
    """Rút chủ đề/lĩnh vực người dùng muốn liệt kê, ví dụ 'đất đai', 'kết hôn', 'hộ tịch'."""
    q = _compact_text(raw_query)
    if not q:
        return ""

    # Ưu tiên các mẫu có quan hệ rõ ràng.
    patterns = [
        r"(?:liên quan đến|liên quan tới|lien quan den|lien quan toi)\s+(.+?)(?:\s+không|\s+khong|,|\?|$)",
        r"(?:thuộc lĩnh vực|thuoc linh vuc|lĩnh vực|linh vuc)\s+(.+?)(?:\s+không|\s+khong|,|\?|$)",
        r"(?:về|ve)\s+(.+?)(?:\s+không|\s+khong|,|\?|$)",
        r"(?:danh sách|danh sach|liệt kê|liet ke)\s+(?:các|cac|những|nhung)?\s*(?:thủ tục|thu tuc)?\s*(.+?)(?:\s+đi|\s+di|\s+nhé|\s+nhe|,|\?|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            topic = _clean_list_topic_text(m.group(1))
            if topic:
                return topic

    # Fallback: bỏ lệnh và giữ lại phần có nghĩa.
    return _clean_list_topic_text(q)


def _list_query_terms(raw_query: str):
    """Lấy token chủ đề để lọc danh sách. Chỉ dùng token nguyên vẹn, không match substring."""
    topic = _extract_related_list_topic(raw_query)
    source = topic or raw_query
    tokens = _meaningful_tokens(source)
    stop = {
        "danh", "sach", "liet", "ke", "thu", "tuc", "nao", "khac", "lien", "quan",
        "linh", "vuc", "bao", "nhieu", "cho", "xem", "duoc", "khong", "toi", "minh",
        "cac", "nhung", "di", "nhe", "ve", "den", "toi", "co", "con",
    }
    return [t for t in tokens if t not in stop]


def _norm_token_set(text: str):
    return set(_meaningful_tokens(text or ""))


def _contains_norm_phrase(text_norm: str, phrase_norm: str) -> bool:
    """Match cụm theo ranh giới từ, tránh 'hon' khớp nhầm trong 'khong' hoặc 'thon'."""
    if not text_norm or not phrase_norm:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase_norm) + r"(?![a-z0-9])", text_norm) is not None


def _procedure_list_match_score(raw_query: str, item: dict) -> float:
    """
    Chấm điểm thủ tục khi người dùng hỏi dạng danh sách theo lĩnh vực/chủ đề.

    Bản phòng thủ:
    - Không dùng substring từng từ, vì 'hôn' -> 'hon' dễ khớp nhầm trong 'không', 'thôn'.
    - Nếu chủ đề có từ ghép như 'kết hôn', 'đất đai', 'hộ tịch' thì yêu cầu khớp cả cụm hoặc đủ token.
    - Ưu tiên mạnh tên thủ tục và lĩnh vực; cơ quan/kết quả chỉ là tín hiệu phụ.
    """
    content = item.get("content") or {}
    name = item.get("name") or content.get("Tên thủ tục") or ""
    field = content.get("Lĩnh vực") or ""
    agency = content.get("Cơ quan thực hiện") or ""
    result = content.get("Kết quả thực hiện") or ""

    topic = _extract_related_list_topic(raw_query)
    topic_norm = _norm_for_match(topic)
    query_terms = list(dict.fromkeys(_list_query_terms(raw_query)))
    if not topic_norm and not query_terms:
        return 0.0

    name_norm = _norm_for_match(name)
    field_norm = _norm_for_match(field)
    agency_norm = _norm_for_match(agency)
    result_norm = _norm_for_match(result)

    # Tránh lẫn nhóm trái nghĩa trong danh sách. Ví dụ hỏi "kết hôn" thì không đưa thủ tục "ly hôn/hủy kết hôn".
    query_norm = _norm_for_match(raw_query)
    if _contains_norm_phrase(topic_norm, "ket hon"):
        if (_contains_norm_phrase(name_norm, "ly hon") or _contains_norm_phrase(name_norm, "huy viec ket hon")) and not (
            _contains_norm_phrase(query_norm, "ly hon") or _contains_norm_phrase(query_norm, "huy viec ket hon")
        ):
            return 0.0

    name_tokens = _norm_token_set(name)
    field_tokens = _norm_token_set(field)
    agency_tokens = _norm_token_set(agency)
    result_tokens = _norm_token_set(result)
    all_tokens = name_tokens | field_tokens | agency_tokens | result_tokens

    score = 0.0
    strong_phrase_match = False

    if topic_norm:
        if topic_norm == field_norm:
            score += 260
            strong_phrase_match = True
        elif _contains_norm_phrase(field_norm, topic_norm) or _contains_norm_phrase(topic_norm, field_norm):
            score += 210
            strong_phrase_match = True
        elif _contains_norm_phrase(name_norm, topic_norm):
            score += 190
            strong_phrase_match = True
        elif _contains_norm_phrase(result_norm, topic_norm):
            score += 45
        elif _contains_norm_phrase(agency_norm, topic_norm):
            score += 18

    matched_name = 0
    matched_field = 0
    matched_result = 0
    matched_agency = 0
    for term in query_terms:
        if term in field_tokens:
            score += 70
            matched_field += 1
        if term in name_tokens:
            score += 55
            matched_name += 1
        if term in result_tokens:
            score += 12
            matched_result += 1
        if term in agency_tokens:
            score += 4
            matched_agency += 1

    unique_terms = set(query_terms)
    matched_terms = len(unique_terms & all_tokens)

    # Nếu chủ đề là cụm nhiều từ, không cho lọt thủ tục chỉ khớp 1 từ.
    # Đây là lỗi gây ra 949 kết quả khi hỏi 'kết hôn': token 'hon' khớp nhầm trong 'khong/thôn'.
    if len(unique_terms) >= 2:
        if not strong_phrase_match and matched_terms < min(2, len(unique_terms)):
            return 0.0

    if len(unique_terms) == 1 and matched_terms == 0 and not strong_phrase_match:
        return 0.0

    # Ưu tiên những thủ tục có chủ đề nằm trong tên/lĩnh vực, không phải chỉ ở cơ quan/kết quả.
    if not strong_phrase_match and matched_name == 0 and matched_field == 0:
        score -= 40

    return max(score, 0.0)


def _find_related_procedures_for_list(raw_query: str, selected_proc_name: str = "", limit: int = 12):
    """
    Tìm danh sách thủ tục từ JSON local, không dùng Qdrant/LLM.
    Dùng cho câu hỏi dạng: liệt kê các thủ tục liên quan đến X.
    """
    q = _compact_text(raw_query)
    exclude_selected = bool(selected_proc_name and _contains_any_phrase(q, ["khác", "khac", "còn", "con"]))

    scored = []
    for item in PROCEDURES_DATA or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or (item.get("content") or {}).get("Tên thủ tục") or ""
        if not name:
            continue
        if exclude_selected and name == selected_proc_name:
            continue

        score = _procedure_list_match_score(raw_query, item)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: (x[0], -len(x[1].get("name", ""))), reverse=True)
    return [item for score, item in scored[:limit]], len(scored)


def _format_related_procedures_answer(raw_query: str, items, total_count: int) -> str:
    topic = _extract_related_list_topic(raw_query)
    topic_display = topic or "nội dung này"

    if not items:
        return (
            f"Mình chưa tìm thấy danh sách thủ tục phù hợp với '{topic_display}' trong dữ liệu hiện có.\n\n"
            "Bạn có thể nhắn rõ hơn theo lĩnh vực hoặc tên giấy tờ cần làm, ví dụ: đất đai, hộ tịch, khuyến mại, hộ kinh doanh."
        )

    lines = [
        f"Mình tìm thấy {total_count} thủ tục có liên quan đến {topic_display}.",
        "Dưới đây là danh sách phù hợp nhất:",
        "",
    ]

    for idx, item in enumerate(items, start=1):
        content = item.get("content") or {}
        name = item.get("name") or content.get("Tên thủ tục") or "Không có tên thủ tục"
        field = content.get("Lĩnh vực") or ""
        agency = content.get("Cơ quan thực hiện") or ""

        lines.append(f"{idx}. {_clean_procedure_display_name(name)}")
        if field:
            lines.append(f"   - Lĩnh vực: {field}")
        if agency:
            lines.append(f"   - Cơ quan giải quyết: {_shorten_for_citizen(agency, max_len=140)}")

    if total_count > len(items):
        lines.append("")
        lines.append(f"Còn {total_count - len(items)} thủ tục khác cũng có liên quan. Bạn có thể nhập tên thủ tục trong danh sách để mình hướng dẫn chi tiết hơn.")
    else:
        lines.append("")
        lines.append("Bạn muốn xem chi tiết thủ tục nào thì nhắn tên thủ tục đó, mình sẽ hướng dẫn tiếp.")

    return _tidy_answer("\n".join(lines))

def _classify_conversation_route(raw_query: str, selected_proc_name: str = "", explicit_proc: str = "", field=None) -> str:
    """
    Phân luồng trước RAG.

    Không có route nào được gọi Qdrant trực tiếp. Route chỉ quyết định:
    - trả lời nhanh;
    - hỏi lại;
    - dùng selected_context;
    - hay cho phép pipeline RAG chạy tiếp.
    """
    if _is_acknowledgement_query(raw_query):
        return ROUTE_ACKNOWLEDGEMENT

    if _is_social_intent_query(raw_query):
        return ROUTE_SOCIAL

    # Câu hỏi yêu cầu danh sách/đếm nhóm thủ tục phải đi riêng.
    # Không để resolver kéo thành một thủ tục đơn lẻ.
    if _is_related_list_query(raw_query):
        return ROUTE_LIST_QUERY

    # Người dùng nêu rõ thủ tục hiện tại hoặc thủ tục mới.
    if explicit_proc:
        if selected_proc_name and explicit_proc != selected_proc_name:
            return ROUTE_SWITCH_PROCEDURE
        return ROUTE_ADMIN_CONFIDENT

    # Không có thủ tục mới, nhưng có câu nối tiếp hợp lệ thì giữ mạch hội thoại.
    if selected_proc_name and _is_valid_context_followup(raw_query, field=field):
        return ROUTE_CONTINUE_CONTEXT

    # Có vẻ thuộc miền thủ tục hành chính nhưng chưa rõ thủ tục nào.
    if _query_has_admin_positive_evidence(raw_query, field=field, explicit_proc=explicit_proc):
        return ROUTE_ASK_CLARIFY

    # Không có bằng chứng hành chính, không có nối tiếp hợp lệ.
    return ROUTE_OUT_OF_DOMAIN


def _build_out_of_domain_answer() -> str:
    return (
        "Mình chủ yếu hỗ trợ tra cứu thủ tục hành chính ạ.\n\n"
        "Nếu bạn cần hỏi về giấy tờ, hồ sơ, nơi nộp, thời hạn hoặc phí/lệ phí, "
        "bạn nhắn tên thủ tục hoặc tên giấy tờ cần làm, mình sẽ hướng dẫn tiếp."
    )


def _build_clarify_answer(selected_proc_name: str = "") -> str:
    if selected_proc_name:
        display_name = _clean_procedure_display_name(selected_proc_name)
        return (
            f"Mình chưa chắc bạn muốn hỏi tiếp thủ tục {display_name} hay muốn chuyển sang thủ tục khác.\n\n"
            "Bạn nhắn rõ hơn giúp mình nhé. Ví dụ: hồ sơ gồm gì, nộp ở đâu, bao lâu có kết quả, "
            "hoặc ghi tên thủ tục mới cần hỏi."
        )

    return (
        "Mình chưa xác định chắc bạn muốn hỏi thủ tục nào.\n\n"
        "Bạn cho mình biết tên giấy tờ hoặc việc cần làm nhé. Ví dụ: đăng ký khai sinh, "
        "đăng ký kết hôn, xin giấy xác nhận tình trạng hôn nhân."
    )

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


# ===== PROCEDURE LOCK / SOURCE GUARD HELPERS =====
def _normalize_procedure_id(value) -> str:
    return _clean_proc_id_value(value)


def _same_procedure_id(left, right) -> bool:
    """
    So khớp mã thủ tục an toàn hơn phép ==.
    Cho phép mã rút gọn 2.000575 khớp với mã đầy đủ 2.000575.000.00.00.H35.
    """
    left = _normalize_procedure_id(left)
    right = _normalize_procedure_id(right)
    if not left or not right:
        return False
    if left == right:
        return True

    left_aliases = set(_procedure_id_aliases(left))
    right_aliases = set(_procedure_id_aliases(right))
    if left_aliases & right_aliases:
        return True

    return left.startswith(right + ".") or right.startswith(left + ".")


def _normalize_procedure_name_for_compare(value: str) -> str:
    value = re.sub(r"(?i)^thủ tục\s+", "", value or "").strip()
    return _norm_for_match(value)


def _same_procedure_name(left: str, right: str) -> bool:
    left_norm = _normalize_procedure_name_for_compare(left)
    right_norm = _normalize_procedure_name_for_compare(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    # Cho phép tên request là phần đầu của tên đầy đủ trong dữ liệu,
    # ví dụ: "Cấp lại Giấy chứng nhận đăng ký hộ kinh doanh"
    # khớp với "Cấp lại..., Cấp đổi sang...".
    shorter, longer = sorted([left_norm, right_norm], key=len)
    return len(shorter) >= 12 and shorter in longer


def _resolve_procedure_name_by_id(procedure_id: str) -> str:
    proc_id = _normalize_procedure_id(procedure_id)
    if not proc_id:
        return ""

    for alias in _procedure_id_aliases(proc_id):
        name = PROCEDURE_NAME_BY_ID.get(alias)
        if name:
            return name

    for known_id, name in PROCEDURE_NAME_BY_ID.items():
        if _same_procedure_id(proc_id, known_id):
            return name

    return ""


def _resolve_locked_procedure(procedure_id: str = "", procedure_name: str = ""):
    """
    Chuẩn hóa khóa thủ tục từ request/frontend hoặc session.
    Trả về (procedure_id, procedure_name). Nếu chỉ có tên thì cố gắng lấy mã;
    nếu chỉ có mã thì cố gắng lấy tên từ procedures.json.
    """
    proc_id = _normalize_procedure_id(procedure_id)
    proc_name = (procedure_name or "").strip()

    # Nếu có mã thì thử lấy tên chuẩn từ dữ liệu trước. Mã request đáng tin hơn resolver.
    name_from_id = _resolve_procedure_name_by_id(proc_id) if proc_id else ""

    if proc_name:
        proc_name = _resolve_proc_name(proc_name) or proc_name

    if name_from_id:
        # Tuyệt đối không để name_from_id ghi đè proc_name nếu ID bị trùng giữa các thủ tục
        if not proc_name:
            proc_name = name_from_id
        elif _same_procedure_name(proc_name, name_from_id):
            proc_name = name_from_id

    if proc_id and not proc_name:
        proc_name = name_from_id

    if proc_name and not proc_id:
        proc_id = _normalize_procedure_id(PROCEDURE_ID_BY_NAME.get(proc_name, ""))

    return proc_id, proc_name


def _get_metadata_procedure_id(meta: dict) -> str:
    meta = meta or {}
    return _normalize_procedure_id(
        meta.get("procedure_id")
        or meta.get("id")
        or meta.get("search_code")
        or meta.get("procedure_code")
    )


def _get_metadata_procedure_name(meta: dict) -> str:
    meta = meta or {}
    return meta.get("procedure_name") or meta.get("name") or ""


def _doc_matches_locked_procedure(doc, procedure_name: str = "", procedure_id: str = "") -> bool:
    """
    Chốt chặn nguồn: doc chỉ hợp lệ nếu thuộc đúng thủ tục đã khóa.
    Ưu tiên so bằng procedure_id/id; nếu metadata thiếu id thì fallback sang name.
    """
    if not procedure_id and not procedure_name:
        return True

    meta = getattr(doc, "metadata", {}) or {}
    doc_id = _get_metadata_procedure_id(meta)
    doc_name = _get_metadata_procedure_name(meta)

    match_id = bool(procedure_id and doc_id and _same_procedure_id(doc_id, procedure_id))
    match_name = bool(procedure_name and doc_name and _same_procedure_name(doc_name, procedure_name))

    if procedure_name:
        if match_name:
            return True
        # Chặn đứng chunk có tên hoàn toàn khác lọt vào do trùng ID
        if match_id and not doc_name:
            return True
        return False

    if procedure_id and match_id:
        return True

    return False


def _guard_docs_by_locked_procedure(docs, procedure_name: str = "", procedure_id: str = "", stage: str = ""):
    if not docs or (not procedure_name and not procedure_id):
        return docs or []

    filtered = [
        doc for doc in docs
        if _doc_matches_locked_procedure(doc, procedure_name=procedure_name, procedure_id=procedure_id)
    ]

    removed = len(docs) - len(filtered)
    if removed:
        print(
            f"[SOURCE GUARD] stage={stage or 'unknown'} removed={removed} "
            f"locked_id={procedure_id or None} locked_name={procedure_name or None}"
        )

    return filtered


def _procedure_filter_candidates(procedure_name: str = "", procedure_id: str = "", section_type: str = ""):
    """
    Qdrant payload ở các lần build có thể dùng metadata.procedure_id hoặc metadata.id.
    Hàm này thử theo thứ tự an toàn: procedure_id -> id -> name.
    """
    candidates = []
    proc_id = _normalize_procedure_id(procedure_id)

    def add(conditions):
        clean = {k: v for k, v in conditions.items() if v}
        if clean and clean not in candidates:
            candidates.append(clean)

    if proc_id:
        for alias in _procedure_id_aliases(proc_id):
            add({"procedure_id": alias, "section_type": section_type})
            add({"id": alias, "section_type": section_type})
            add({"search_code": alias, "section_type": section_type})
            add({"procedure_code": alias, "section_type": section_type})

    if procedure_name:
        add({"name": procedure_name, "section_type": section_type})
        add({"procedure_name": procedure_name, "section_type": section_type})

    return candidates


def _scroll_docs_by_procedure(db, procedure_name: str = "", procedure_id: str = "", limit: int = 80, section_type: str = ""):
    """Lấy docs theo thủ tục đã khóa, ưu tiên mã thủ tục để tránh nhầm tên."""
    for conditions in _procedure_filter_candidates(procedure_name, procedure_id, section_type):
        qdrant_filter = build_qdrant_filter(**conditions)
        docs = _scroll_docs_by_filter(db=db, qdrant_filter=qdrant_filter, limit=limit)
        docs = _guard_docs_by_locked_procedure(
            docs,
            procedure_name=procedure_name,
            procedure_id=procedure_id,
            stage=f"scroll:{conditions}",
        )
        if docs:
            return docs
    return []


def _similarity_search_by_procedure(db, query: str, k: int, procedure_name: str = "", procedure_id: str = "", section_type: str = ""):
    """Vector search nhưng vẫn bị khóa trong đúng thủ tục."""
    for conditions in _procedure_filter_candidates(procedure_name, procedure_id, section_type=section_type):
        qdrant_filter = build_qdrant_filter(**conditions)
        try:
            docs = db.similarity_search(query, k=k, filter=qdrant_filter)
        except Exception as e:
            print(f"[PROC FILTER VECTOR ERROR] conditions={conditions}: {e}")
            docs = []

        docs = _guard_docs_by_locked_procedure(
            docs,
            procedure_name=procedure_name,
            procedure_id=procedure_id,
            stage=f"vector:{conditions}",
        )
        if docs:
            return docs
    return []



# ===== SPEED HELPERS =====
def _has_context_reference(query: str) -> bool:
    """
    Nhận diện câu hỏi phụ thuộc ngữ cảnh: "vậy", "còn thủ tục này", "việc này", "cái này", "nó", ...
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

    # Nếu câu hiện tại đã xác định được thủ tục + field, hoặc có dấu hiệu nêu thủ tục mới,
    # không rewrite bằng LLM nữa. Rewrite rất dễ kéo lịch sử cũ vào và làm sai hướng.
    if detected_proc and (detected_field or _has_explicit_procedure_signal(raw_query)):
        return False

    if not detected_proc:
        return True

    has_history = bool(history)
    if has_history and _has_context_reference(raw_query):
        return True

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
        "cộng thêm",
        "cộng vào",
        "nhân với",
        "chia cho",
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
    Hỗ trợ cả header cũ:
        Thủ tục / Lĩnh vực / Loại thông tin
    và header Markdown mới:
        # Thủ tục / - Mã thủ tục / - Lĩnh vực / - Mục thông tin / ## Section
    """
    lines = []
    skipped_first_section_heading = False

    for line in chunk_text.splitlines():
        stripped = line.strip()

        if stripped.startswith("Thủ tục:"):
            continue
        if stripped.startswith("Lĩnh vực:"):
            continue
        if stripped.startswith("Loại thông tin:"):
            continue
        if stripped.startswith("# Thủ tục:"):
            continue
        if stripped.startswith("- Mã thủ tục:"):
            continue
        if stripped.startswith("- Lĩnh vực:"):
            continue
        if stripped.startswith("- Mục thông tin:"):
            continue

        # Bỏ heading section đầu tiên của child chunk để parser nhãn bên dưới hoạt động như cũ.
        if stripped.startswith("## ") and not skipped_first_section_heading:
            skipped_first_section_heading = True
            continue

        lines.append(line)

    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text



def _extract_chunk_section(chunk_text: str) -> str:
    """
    Lấy tên field từ header chunk.
    Hỗ trợ cả 'Loại thông tin:' cũ và '- Mục thông tin:' Markdown mới.
    """
    match = re.search(r"(?m)^Loại thông tin:\s*(.+?)\s*$", chunk_text)
    if match:
        return match.group(1).strip()

    match = re.search(r"(?m)^-\s*Mục thông tin:\s*(.+?)\s*$", chunk_text)
    if match:
        return match.group(1).strip()

    return ""



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


def _friendly_proc_name(name: str) -> str:
    return _clean_procedure_display_name(name or "") or (name or "thủ tục này")


def _tidy_answer(answer: str) -> str:
    answer = (answer or "").replace("**", "")
    answer = re.sub(r"(?m)^\s*\*\s+", "- ", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer


def _shorten_for_citizen(text: str, max_len: int = 320) -> str:
    """
    Làm câu chữ bớt ngợp cho người dân, nhưng vẫn giữ ý chính.
    Không dùng để thay đổi dữ liệu pháp lý, chỉ cắt hiển thị quá dài.
    """
    value = re.sub(r"\s+", " ", (text or "")).strip()
    if len(value) <= max_len:
        return value

    cut = value[:max_len]
    last_stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(". "))
    if last_stop > 120:
        return cut[:last_stop + 1].strip() + "..."
    return cut.rstrip() + "..."


def _friendly_no_data(label: str, detected_proc: str) -> str:
    proc = _friendly_proc_name(detected_proc)
    return (
        f"Hiện dữ liệu chưa ghi thông tin {label} cho thủ tục {proc}.\n\n"
        "Bạn có thể hỏi tiếp phần hồ sơ, nơi nộp hoặc thời hạn. Khi đi nộp hồ sơ, bạn nên hỏi lại bộ phận một cửa để chắc chắn nhất."
    )


def _format_document_direct_answer(detected_proc: str, context_chunks, raw_query: str = ""):
    """
    Format riêng cho câu hỏi hồ sơ/giấy tờ theo lối dễ đọc cho người dân.

    Bản v2 phòng thủ dữ liệu mới:
    - Bỏ các dòng "Lưu ý", "Trường hợp...", số lượng 0 bản chính khỏi danh sách giấy tờ chính.
    - Nếu người dùng hỏi ra xã/trực tiếp thì không ưu tiên giấy tờ chỉ dành cho nộp trực tuyến.
    - Chỉ dùng các mục bị lọc khi thật sự không tìm thấy giấy tờ chính nào.
    """
    main_items = []
    fallback_items = []
    seen = set()
    direct_submit = _is_direct_submit_query(raw_query, detected_proc)

    for chunk in context_chunks:
        body = _strip_chunk_header(chunk)
        nhom_ho_so = _extract_label_block(body, "Nhóm hồ sơ:")
        ten_giay_to = _extract_label_block(body, "Tên giấy tờ:")
        so_luong = _extract_label_block(body, "Số lượng:")

        if not ten_giay_to:
            continue

        key = re.sub(r"\s+", " ", ten_giay_to.lower()).strip()
        if key in seen:
            continue
        seen.add(key)

        item_lines = [_shorten_for_citizen(ten_giay_to, max_len=420)]
        if so_luong and not so_luong.strip().lower().startswith("0"):
            item_lines.append(f"Số lượng: {_shorten_for_citizen(so_luong, max_len=160)}")
        item = "\n".join(item_lines)

        is_note = _is_document_note_or_non_required(ten_giay_to, so_luong, nhom_ho_so)
        is_online_only = direct_submit and _is_online_only_document_text(" ".join([nhom_ho_so, ten_giay_to]))

        if is_note or is_online_only:
            fallback_items.append(item)
            continue

        main_items.append(item)
        if len(main_items) >= DIRECT_ANSWER_MAX_CHUNKS:
            break

    items = main_items or fallback_items[:DIRECT_ANSWER_MAX_CHUNKS]
    if not items:
        return None

    proc = _friendly_proc_name(detected_proc)
    lines = [
        f"Với thủ tục {proc}, bạn chuẩn bị các giấy tờ chính sau:",
        "",
    ]

    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item}")
        lines.append("")

    if main_items and fallback_items:
        lines.append("Một số lưu ý hoặc trường hợp đặc biệt có thể phát sinh, nhưng các mục trên là phần giấy tờ chính cần chuẩn bị trước.")
    else:
        lines.append("Nếu bạn muốn, tôi có thể chỉ tiếp nơi nộp hồ sơ hoặc thời hạn giải quyết.")
    return _tidy_answer("\n".join(lines))

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



def _as_field_list(field):
    if not field:
        return []
    if isinstance(field, str):
        return [field]
    if isinstance(field, list):
        return field
    return list(field)


def _is_condition_field(field) -> bool:
    if not field:
        return False
    fields_list = [str(f).lower() for f in _as_field_list(field)]
    return any("yêu cầu điều kiện" in f or "điều kiện" in f for f in fields_list)


def _is_direct_submit_query(raw_query: str, procedure_name: str = "") -> bool:
    q = _strip_procedure_mention_for_intent(raw_query, procedure_name)
    return any(x in q for x in [
        "ra xã", "lên xã", "đến xã", "tại xã", "trực tiếp", "mang",
        "đem", "nộp tại", "nộp ở", "đến ubnd", "ủy ban", "uỷ ban",
    ])


def _is_online_only_document_text(text: str) -> bool:
    q = (text or "").lower()

    # Mẫu điện tử tương tác chỉ dành cho nộp trực tuyến.
    if "mẫu hộ tịch điện tử" in q or "mau ho tich dien tu" in q:
        return True

    has_online = any(x in q for x in [
        "trực tuyến", "truc tuyen", "cổng dịch vụ công", "cong dich vu cong",
        "tải lên", "tai len",
    ])
    has_direct = any(x in q for x in [
        "trực tiếp", "truc tiep", "xuất trình", "xuat trinh", "mang", "nộp/xuất trình", "nop/xuat trinh",
    ])

    # Nếu một giấy tờ ghi cả trực tiếp và trực tuyến thì vẫn giữ khi người dân hỏi ra xã,
    # vì nó vẫn là giấy tờ có thể phải xuất trình trực tiếp.
    return has_online and not has_direct

def _is_document_note_or_non_required(ten_giay_to: str, so_luong: str = "", nhom_ho_so: str = "") -> bool:
    text = re.sub(r"\s+", " ", (ten_giay_to or "").strip().lower())
    group = re.sub(r"\s+", " ", (nhom_ho_so or "").strip().lower())
    qty = re.sub(r"\s+", " ", (so_luong or "").strip().lower())

    if "lưu ý" in group or "luu y" in group:
        return True
    if qty.startswith("0 ") or qty == "0" or "0 bản" in qty or "0 ban" in qty:
        return True

    note_starts = (
        "trường hợp", "truong hop", "+",
        "người tiếp nhận", "nguoi tiep nhan",
        "đối với", "doi voi", "bản chụp", "ban chup",
        "cá nhân có quyền", "ca nhan co quyen",
        "nếu bên", "neu ben", "người có yêu cầu", "nguoi co yeu cau",
    )
    return text.startswith(note_starts)


def _is_fee_question(raw_query: str, procedure_name: str = "") -> bool:
    q = _strip_procedure_mention_for_intent(raw_query, procedure_name)
    return any(x in q for x in [
        "phí",
        "lệ phí",
        "mất tiền",
        "bao nhiêu tiền",
        "tốn tiền",
        "tốn phí",
        "chi phí",
        "có mất",
        "đóng bao nhiêu",
        "mấy nghìn",
    ])


def _is_time_question(raw_query: str, procedure_name: str = "") -> bool:
    q = _strip_procedure_mention_for_intent(raw_query, procedure_name)
    return any(x in q for x in [
        "bao lâu",
        "mất bao lâu",
        "thời hạn",
        "thời gian",
        "mấy ngày",
        "khi nào xong",
        "giải quyết trong bao lâu",
        "chờ bao lâu",
        "xong trong",
    ])


def _is_location_question(raw_query: str, procedure_name: str = "") -> bool:
    q = _strip_procedure_mention_for_intent(raw_query, procedure_name)
    return any(x in q for x in [
        "ở đâu",
        "nộp ở đâu",
        "nộp tại đâu",
        "nộp chỗ nào",
        "cơ quan nào",
        "cơ quan thực hiện",
        "đến đâu",
        "làm ở đâu",
        "ra xã",
        "lên huyện",
        "tại đâu",
    ])


def _is_method_question(raw_query: str, procedure_name: str = "") -> bool:
    q = _strip_procedure_mention_for_intent(raw_query, procedure_name)
    return any(x in q for x in [
        "cách thức",
        "hình thức",
        "cách nộp",
        "nộp online",
        "nộp trực tuyến",
        "trực tuyến",
        "online",
        "dịch vụ công",
        "bưu chính",
        "qua bưu điện",
        "nộp qua mạng",
    ])


def _is_document_question(raw_query: str, procedure_name: str = "") -> bool:
    q = _strip_procedure_mention_for_intent(raw_query, procedure_name)
    return any(x in q for x in [
        "hồ sơ",
        "giấy tờ",
        "cần chuẩn bị",
        "chuẩn bị gì",
        "cần mang",
        "thành phần hồ sơ",
        "cần những gì",
        "giấy nào",
    ])


def _format_fee_only_direct_answer(detected_proc: str, context_chunks):
    """
    Format riêng cho câu hỏi phí/lệ phí. Không kéo theo thời hạn/cách thức.
    """
    fee_lines = []
    seen = set()

    for chunk in context_chunks:
        section = _extract_chunk_section(chunk)
        body = _strip_chunk_header(chunk).strip()
        if not body:
            continue

        value = ""
        method = _compact_method_name(_extract_label_block(body, "Hình thức:"))
        method_fee = _clean_fee_text(_extract_label_block(body, "Mức phí/Lệ phí:"))
        if method_fee:
            value = f"{method}: {method_fee}" if method else method_fee
        elif section in ["Phí", "Lệ phí"]:
            value = _clean_fee_text(body) or body

        value = _shorten_for_citizen(value, max_len=280)
        if not value:
            continue

        key = re.sub(r"\s+", " ", value.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        fee_lines.append(value)

    if not fee_lines:
        return _friendly_no_data("phí/lệ phí", detected_proc)

    proc = _friendly_proc_name(detected_proc)
    lines = [f"Về phí/lệ phí của thủ tục {proc}:", ""]
    for item in fee_lines[:DIRECT_ANSWER_MAX_CHUNKS]:
        lines.append(f"- {item}")
    return _tidy_answer("\n".join(lines))


def _format_time_only_direct_answer(detected_proc: str, context_chunks):
    """
    Format riêng cho câu hỏi thời hạn. Không kéo theo phí/lệ phí.
    """
    time_lines = []
    seen = set()

    for chunk in context_chunks:
        section = _extract_chunk_section(chunk)
        if section not in ["Thời hạn giải quyết", "Cách thức thực hiện"]:
            continue

        body = _strip_chunk_header(chunk).strip()
        if not body:
            continue

        method = _compact_method_name(_extract_label_block(body, "Hình thức:"))
        deadline = _clean_time_text(_extract_label_block(body, "Thời hạn:"))

        if deadline:
            value = f"{method}: {deadline}" if method else deadline
        elif section == "Thời hạn giải quyết":
            value = _clean_time_text(body)
        else:
            value = ""

        value = _shorten_for_citizen(value, max_len=280)
        if not value:
            continue

        key = re.sub(r"\s+", " ", value.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        time_lines.append(value)

    if not time_lines:
        return _friendly_no_data("thời hạn giải quyết", detected_proc)

    proc = _friendly_proc_name(detected_proc)
    lines = [f"Thời hạn giải quyết thủ tục {proc}:", ""]
    for item in time_lines[:DIRECT_ANSWER_MAX_CHUNKS]:
        lines.append(f"- {item}")
    return _tidy_answer("\n".join(lines))


def _format_agency_result_direct_answer(detected_proc: str, context_chunks):
    """
    Format nhanh cho câu hỏi nơi nộp/cơ quan/kết quả.
    """
    agencies = []
    results = []
    seen_agency = set()
    seen_result = set()

    for chunk in context_chunks:
        section = _extract_chunk_section(chunk)
        body = _strip_chunk_header(chunk).strip()
        if not body:
            continue

        value = _shorten_for_citizen(body, max_len=360)
        key = re.sub(r"\s+", " ", value.lower()).strip()

        if section == "Cơ quan thực hiện" and key not in seen_agency:
            agencies.append(value)
            seen_agency.add(key)
        elif section == "Kết quả thực hiện" and key not in seen_result:
            results.append(value)
            seen_result.add(key)

    if not agencies and not results:
        return None

    proc = _friendly_proc_name(detected_proc)
    lines = [f"Với thủ tục {proc}:", ""]

    if agencies:
        lines.append("Nơi giải quyết:")
        for item in agencies[:DIRECT_ANSWER_MAX_CHUNKS]:
            lines.append(f"- {item}")
        lines.append("")

    if results:
        lines.append("Kết quả nhận được:")
        for item in results[:DIRECT_ANSWER_MAX_CHUNKS]:
            lines.append(f"- {item}")
        lines.append("")

    return _tidy_answer("\n".join(lines))


def _is_result_question(raw_query: str, procedure_name: str = "") -> bool:
    q = _strip_procedure_mention_for_intent(raw_query, procedure_name)
    return any(x in q for x in [
        "kết quả",
        "nhận được gì",
        "được cấp gì",
        "trả về gì",
        "làm xong được gì",
        "làm xong nhận",
        "giấy gì",
    ])


def _format_result_only_direct_answer(detected_proc: str, context_chunks):
    results = []
    seen = set()

    for chunk in context_chunks:
        section = _extract_chunk_section(chunk)
        if section != "Kết quả thực hiện":
            continue

        body = _strip_chunk_header(chunk).strip()
        if not body:
            continue

        value = _shorten_for_citizen(body, max_len=320)
        key = re.sub(r"\s+", " ", value.lower()).strip()
        if key in seen:
            continue

        seen.add(key)
        results.append(value)
        if len(results) >= DIRECT_ANSWER_MAX_CHUNKS:
            break

    if not results:
        return _friendly_no_data("kết quả thực hiện", detected_proc)

    proc = _friendly_proc_name(detected_proc)
    lines = [f"Kết quả nhận được của thủ tục {proc}:", ""]
    for item in results:
        lines.append(f"- {item}")
    return _tidy_answer("\n".join(lines))


def _format_method_only_direct_answer(detected_proc: str, context_chunks, raw_query: str = ""):
    methods = []
    seen = set()

    for chunk in context_chunks:
        section = _extract_chunk_section(chunk)
        if section != "Cách thức thực hiện":
            continue

        body = _strip_chunk_header(chunk).strip()
        if not body:
            continue

        method = _compact_method_name(_extract_label_block(body, "Hình thức:"))
        deadline = _clean_time_text(_extract_label_block(body, "Thời hạn:"))
        fee = _clean_fee_text(_extract_label_block(body, "Mức phí/Lệ phí:"))

        if method:
            value = method
            extra = []
            if deadline and _is_time_question(raw_query, detected_proc):
                extra.append(f"thời hạn: {deadline}")
            if fee and _is_fee_question(raw_query, detected_proc):
                extra.append(f"phí/lệ phí: {fee}")
            if extra:
                value += " (" + "; ".join(extra) + ")"
        else:
            value = _shorten_for_citizen(body, max_len=260)

        key = re.sub(r"\s+", " ", value.lower()).strip()
        if not value or key in seen:
            continue
        seen.add(key)
        methods.append(value)
        if len(methods) >= DIRECT_ANSWER_MAX_CHUNKS:
            break

    if not methods:
        return _friendly_no_data("cách thức thực hiện", detected_proc)

    proc = _friendly_proc_name(detected_proc)
    lines = [f"Cách thực hiện thủ tục {proc}:", ""]
    for item in methods:
        lines.append(f"- {item}")
    return _tidy_answer("\n".join(lines))


_FIELD_GROUP_ORDER = ["document", "agency", "time", "fee", "method", "result", "condition", "legal"]
_FIELD_GROUP_LABELS = {
    "document": "Hồ sơ cần chuẩn bị",
    "agency": "Nơi nộp / cơ quan giải quyết",
    "time": "Thời hạn giải quyết",
    "fee": "Phí/lệ phí",
    "method": "Cách thực hiện",
    "result": "Kết quả nhận được",
    "condition": "Điều kiện cần lưu ý",
    "legal": "Căn cứ pháp lý",
}


def _field_groups_from_query(raw_query: str, field, procedure_name: str = ""):
    fields = set(_as_field_list(field))
    groups = []

    def add(group):
        if group not in groups:
            groups.append(group)

    if "Thành phần hồ sơ" in fields and _is_document_question(raw_query, procedure_name):
        add("document")

    if "Cơ quan thực hiện" in fields and _is_location_question(raw_query, procedure_name):
        add("agency")

    if "Thời hạn giải quyết" in fields and _is_time_question(raw_query, procedure_name):
        add("time")

    if ("Phí" in fields or "Lệ phí" in fields) and _is_fee_question(raw_query, procedure_name):
        add("fee")

    if "Cách thức thực hiện" in fields and _is_method_question(raw_query, procedure_name):
        add("method")

    if "Kết quả thực hiện" in fields and _is_result_question(raw_query, procedure_name):
        add("result")

    if "Yêu cầu điều kiện" in fields:
        add("condition")

    if any(f in fields for f in ["Căn cứ pháp lý", "Cơ quan ban hành", "Cơ quan phối hợp"]):
        add("legal")

    # Nếu câu hỏi ghép nhưng một vài field bị detect mà không bắt được keyword ý định,
    # vẫn giữ lại theo field để tránh trả thiếu. Tuy nhiên không coi Cách thức đi kèm phí/thời hạn là một nhóm riêng.
    if not groups:
        if "Thành phần hồ sơ" in fields:
            add("document")
        if "Cơ quan thực hiện" in fields:
            add("agency")
        if "Thời hạn giải quyết" in fields:
            add("time")
        if "Phí" in fields or "Lệ phí" in fields:
            add("fee")
        if "Kết quả thực hiện" in fields:
            add("result")
        if "Cách thức thực hiện" in fields:
            add("method")

    return [g for g in _FIELD_GROUP_ORDER if g in groups]


def _effective_fields_for_retrieval(raw_query: str, field, procedure_name: str = ""):
    """
    Giảm số field cần lấy trước khi truy xuất Qdrant.

    Lợi ích:
    - Hỏi phí thì không kéo hồ sơ.
    - Hỏi hồ sơ + phí thì chỉ lấy hồ sơ + phí/lệ phí + cách thức.
    - Hỏi ghép vẫn lấy đủ field trong một lần scroll theo thủ tục.
    """
    groups = _field_groups_from_query(raw_query, field, procedure_name)
    if not groups:
        return _as_field_list(field)

    fields = []

    def add(value):
        if value and value not in fields:
            fields.append(value)

    for group in groups[:max(1, MULTI_FIELD_MAX_GROUPS)]:
        if group == "document":
            add("Thành phần hồ sơ")
        elif group == "agency":
            add("Cơ quan thực hiện")
        elif group == "time":
            add("Thời hạn giải quyết")
            add("Cách thức thực hiện")
        elif group == "fee":
            add("Phí")
            add("Lệ phí")
            add("Cách thức thực hiện")
        elif group == "method":
            add("Cách thức thực hiện")
        elif group == "result":
            add("Kết quả thực hiện")
        elif group == "condition":
            add("Yêu cầu điều kiện")
        elif group == "legal":
            add("Căn cứ pháp lý")
            add("Cơ quan ban hành")
            add("Cơ quan phối hợp")

    return fields


def _strip_direct_answer_heading(answer: str) -> str:
    value = _tidy_answer(answer or "")
    if not value:
        return ""

    lines = value.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    if lines:
        first = lines[0].strip()
        intro_patterns = [
            r"^Với thủ tục .+?:$",
            r"^Với thủ tục .+?, bạn chuẩn bị các giấy tờ chính sau:$",
            r"^Về phí/lệ phí của thủ tục .+?:$",
            r"^Thời hạn giải quyết thủ tục .+?:$",
            r"^Kết quả nhận được của thủ tục .+?:$",
            r"^Cách thực hiện thủ tục .+?:$",
            r"^Tôi tìm được thông tin về thủ tục .+?:$",
        ]
        if any(re.match(pattern, first, flags=re.IGNORECASE) for pattern in intro_patterns):
            lines = lines[1:]

    value = "\n".join(lines).strip()
    value = re.sub(r"\n?Nếu bạn muốn, tôi có thể chỉ tiếp.+$", "", value, flags=re.IGNORECASE | re.DOTALL).strip()
    return _tidy_answer(value)


def _build_multi_field_direct_answer(detected_proc: str, field, context_chunks, raw_query: str = ""):
    groups = _field_groups_from_query(raw_query, field, detected_proc)
    if len(groups) < 2:
        return None

    groups = groups[:max(2, MULTI_FIELD_MAX_GROUPS)]
    proc = _friendly_proc_name(detected_proc)
    parts = [f"Với thủ tục {proc}, mình tóm tắt các phần bạn hỏi như sau:", ""]
    index = 1

    for group in groups:
        section_answer = None
        if group == "document":
            section_answer = _format_document_direct_answer(detected_proc, context_chunks, raw_query=raw_query)
        elif group == "agency":
            section_answer = _format_agency_result_direct_answer(detected_proc, context_chunks)
        elif group == "time":
            section_answer = _format_time_only_direct_answer(detected_proc, context_chunks)
        elif group == "fee":
            section_answer = _format_fee_only_direct_answer(detected_proc, context_chunks)
        elif group == "method":
            section_answer = _format_method_only_direct_answer(detected_proc, context_chunks, raw_query=raw_query)
        elif group == "result":
            section_answer = _format_result_only_direct_answer(detected_proc, context_chunks)

        section_body = _strip_direct_answer_heading(section_answer or "")
        if not section_body:
            continue

        parts.append(f"{index}. {_FIELD_GROUP_LABELS.get(group, 'Thông tin')}: ")
        parts.append(section_body)
        parts.append("")
        index += 1

    if index == 1:
        return None

    if len(groups) >= MULTI_FIELD_MAX_GROUPS:
        parts.append("Nếu cần thêm phần khác, bạn cứ hỏi tiếp, mình sẽ chỉ từng phần cho dễ theo dõi.")

    return _tidy_answer("\n".join(parts))


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

    # XỬ LÝ LỖI SAI_06: Loại bỏ "có phải" ra khỏi danh sách kích hoạt bẫy trực tiếp.
    # Để LLM xử lý các câu hỏi "có phải ... không" hoặc "bắt buộc không" để đảm bảo tính logic và ngữ cảnh.
    trap_markers = [
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

    base_answer = _build_direct_answer(detected_proc, field, context_chunks, raw_query=raw_query)
    if not base_answer:
        return None

    if any(m in q for m in ["bỏ qua", "tự bịa", "bịa"]):
        prefix = (
            "Không thể bỏ qua dữ liệu hoặc tự bịa thông tin thủ tục. "
            "Thông tin đúng theo dữ liệu hiện có là:\n\n"
        )
        return prefix + base_answer

    math_markers = ["lấy ngày", "ngày ban hành", "cộng thêm", "cộng vào", "nhân với", "chia cho", "tính toán"]
    if any(m in q for m in math_markers):
        prefix = (
            "Không được tính thời hạn giải quyết bằng cách lấy số hiệu, ngày ban hành "
            "hoặc thông tin văn bản pháp lý để cộng trừ. Thời hạn phải lấy từ mục "
            "thời hạn giải quyết của thủ tục. Thông tin đúng là:\n\n"
        )
        return prefix + base_answer

    return None


def _format_generic_direct_answer(detected_proc: str, context_chunks):
    selected = []
    seen = set()

    for chunk in context_chunks:
        body = _strip_chunk_header(chunk)
        body = _shorten_for_citizen(body, max_len=420)
        if not body or body in seen:
            continue

        selected.append(body)
        seen.add(body)
        if len(selected) >= min(DIRECT_ANSWER_MAX_CHUNKS, 3):
            break

    if not selected:
        return None

    proc = _friendly_proc_name(detected_proc)
    answer = f"Tôi tìm được thông tin về thủ tục {proc}:\n\n" + "\n\n".join(selected)
    return _tidy_answer(answer)


def _build_direct_answer(detected_proc: str, field, context_chunks, raw_query: str = ""):
    """
    Trả lời nhanh từ context, không gọi LLM.
    Ưu tiên theo ý định thật của câu hỏi, không theo thứ tự field bị detect.
    Nhờ vậy:
    - Hỏi phí/lệ phí thì chỉ trả phí/lệ phí.
    - Hỏi thời hạn thì chỉ trả thời hạn.
    - Hỏi nộp ở đâu thì ưu tiên cơ quan thực hiện.
    - Hỏi hồ sơ thì chỉ trả hồ sơ.
    """
    if not detected_proc or not field or not context_chunks:
        return None

    fields = _as_field_list(field)

    # 0. Câu hỏi ghép nhiều ý: trả lời đủ từng mục trong một lần, không gọi LLM.
    multi_answer = _build_multi_field_direct_answer(
        detected_proc=detected_proc,
        field=field,
        context_chunks=context_chunks,
        raw_query=raw_query,
    )
    if multi_answer:
        return multi_answer

    # 1. Câu hỏi nơi nộp/cơ quan phải ưu tiên cơ quan, dù detect_field có lẫn hồ sơ.
    if _is_location_question(raw_query, detected_proc):
        answer = _format_agency_result_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    # 2. Câu hỏi hồ sơ thì chỉ trả hồ sơ.
    if _is_document_question(raw_query, detected_proc):
        answer = _format_document_direct_answer(detected_proc, context_chunks, raw_query=raw_query)
        if answer:
            return answer

    # 3. Câu hỏi phí/lệ phí thì chỉ trả phí/lệ phí.
    if _is_fee_question(raw_query, detected_proc):
        answer = _format_fee_only_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    # 4. Câu hỏi thời hạn thì chỉ trả thời hạn.
    if _is_time_question(raw_query, detected_proc):
        answer = _format_time_only_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    # 4.1. Câu hỏi cách thức/nộp trực tuyến thì chỉ trả cách thức.
    if _is_method_question(raw_query, detected_proc):
        answer = _format_method_only_direct_answer(detected_proc, context_chunks, raw_query=raw_query)
        if answer:
            return answer

    # 4.2. Câu hỏi kết quả thì chỉ trả kết quả.
    if _is_result_question(raw_query, detected_proc):
        answer = _format_result_only_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    # 4.3. Câu hỏi điều kiện/bẫy pháp lý không dùng generic direct answer.
    # Nếu field điều kiện rỗng hoặc chỉ có parent context, phải để LLM đọc toàn thủ tục
    # và trả lời đúng mức dữ liệu cho phép, tránh tóm tắt lan man cả thủ tục.
    if _is_condition_field(fields):
        return None

    # 5. Fallback theo field nếu câu hỏi không rõ ý định.
    if "Cơ quan thực hiện" in fields:
        answer = _format_agency_result_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    if "Kết quả thực hiện" in fields:
        answer = _format_result_only_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    if "Thành phần hồ sơ" in fields:
        answer = _format_document_direct_answer(detected_proc, context_chunks, raw_query=raw_query)
        if answer:
            return answer

    if "Phí" in fields or "Lệ phí" in fields:
        answer = _format_fee_only_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    if "Cách thức thực hiện" in fields and "Thời hạn giải quyết" not in fields:
        answer = _format_method_only_direct_answer(detected_proc, context_chunks, raw_query=raw_query)
        if answer:
            return answer

    if "Thời hạn giải quyết" in fields or "Cách thức thực hiện" in fields:
        answer = _format_time_only_direct_answer(detected_proc, context_chunks)
        if answer:
            return answer

    return None



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


def _exact_field_retrieval(db, query: str, detected_proc: str, field, detected_proc_id: str = ""):
    """
    Fast path tối ưu: đã biết thủ tục + field thì lấy trực tiếp bằng Qdrant scroll.

    Bản procedure-lock:
    - Ưu tiên khóa bằng procedure_id/id thay vì tên thủ tục.
    - Scroll theo đúng thủ tục rồi lọc field trong Python.
    - Fallback vector search vẫn bị khóa trong đúng thủ tục.
    - Source guard loại mọi chunk khác thủ tục trước khi trả lời.
    """
    if not detected_proc and not detected_proc_id:
        return []
    if not field:
        return []

    started = time.perf_counter()
    fields = _effective_fields_for_retrieval(query, field, detected_proc)
    fields = [f for f in _as_field_list(fields) if f]
    if not fields:
        return []

    wanted_fields = set(fields)
    scan_limit = max(EXACT_PROC_SCROLL_LIMIT, EXACT_FIELD_K * max(1, len(wanted_fields)) + 12)

    proc_docs = _scroll_docs_by_procedure(
        db=db,
        procedure_name=detected_proc,
        procedure_id=detected_proc_id,
        limit=scan_limit,
    )

    exact_docs = [
        doc for doc in proc_docs
        if (getattr(doc, "metadata", {}) or {}).get("field") in wanted_fields
    ]

    # Fallback an toàn: nếu scroll không lấy được field nào thì vector search một lần trong đúng thủ tục.
    if not exact_docs:
        fallback_started = time.perf_counter()
        fallback_docs = _similarity_search_by_procedure(
            db=db,
            query=query,
            k=min(scan_limit, max(12, EXACT_FIELD_K * max(1, len(wanted_fields)))),
            procedure_name=detected_proc,
            procedure_id=detected_proc_id,
        )
        fallback_ms = int((time.perf_counter() - fallback_started) * 1000)
        print(f"[FAST EXACT FALLBACK VECTOR ONCE]: fields={fields} latency_ms={fallback_ms}")
        exact_docs = [
            doc for doc in fallback_docs
            if (getattr(doc, "metadata", {}) or {}).get("field") in wanted_fields
        ]

    if not exact_docs:
        return []

    exact_docs = _guard_docs_by_locked_procedure(
        exact_docs,
        procedure_name=detected_proc,
        procedure_id=detected_proc_id,
        stage="exact_field_final",
    )
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
        f"[FAST EXACT PROC SCROLL]: proc='{detected_proc}' proc_id='{detected_proc_id}' "
        f"fields={fields} chunks={len(unique_docs)} latency_ms={elapsed_ms}"
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



# ===== SESSION CONTEXT / RESPONSE HELPERS =====

def _get_proc_id_by_name(procedure_name: str, docs=None) -> str:
    if docs:
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            doc_id = _get_metadata_procedure_id(meta)
            if doc_id:
                return doc_id

    return _normalize_procedure_id(PROCEDURE_ID_BY_NAME.get(procedure_name, ""))


def _build_sources(docs, limit: int = 5):
    sources = []
    seen = set()

    for doc in docs or []:
        meta = getattr(doc, "metadata", {}) or {}
        key = (
            meta.get("procedure_id") or meta.get("id"),
            meta.get("procedure_name") or meta.get("name"),
            meta.get("field"),
            meta.get("chunk_id"),
        )
        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "procedure_id": str(meta.get("procedure_id") or meta.get("id") or ""),
            "procedure_name": meta.get("procedure_name") or meta.get("name") or "",
            "field": meta.get("field") or "",
            "section_type": meta.get("section_type") or "",
            "chunk_id": meta.get("chunk_id") or "",
        })

        if len(sources) >= limit:
            break

    return sources


def _normalize_field_list(field):
    if not field:
        return []
    if isinstance(field, list):
        return field
    return [field]



def _clean_procedure_display_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"(?i)^thủ tục\s+", "", name).strip()
    return name


def build_suggested_questions(procedure_name: str = "", field=None):
    asked_fields = set(_normalize_field_list(field))

    # Chỉ gợi ý khi đã xác định chắc thủ tục.
    # Nếu chưa chắc thủ tục hoặc câu ngoài phạm vi, không sinh gợi ý để tránh kéo người dùng đi sai hướng.
    if not procedure_name:
        return []

    display_name = _clean_procedure_display_name(procedure_name)

    def is_asked(fields_to_check):
        return any(f in asked_fields for f in fields_to_check)

    primary_pool = [
        (["Thành phần hồ sơ"], f"Hồ sơ cần chuẩn bị cho thủ tục {display_name} gồm những gì?"),
        (["Thời hạn giải quyết"], f"Thời hạn giải quyết thủ tục {display_name} là bao lâu?"),
        (["Phí", "Lệ phí"], f"Thủ tục {display_name} có mất phí hoặc lệ phí không?"),
        (["Cơ quan thực hiện"], f"Tôi cần nộp thủ tục {display_name} ở đâu?"),
    ]

    secondary_pool = [
        (["Cách thức thực hiện"], f"Thủ tục {display_name} có thể nộp trực tuyến không?"),
        (["Kết quả thực hiện"], f"Làm xong thủ tục {display_name} sẽ nhận được kết quả gì?"),
        (["Yêu cầu điều kiện"], f"Cần đáp ứng điều kiện gì để làm thủ tục {display_name}?"),
        (["Trình tự thực hiện"], f"Trình tự các bước giải quyết thủ tục {display_name} như thế nào?"),
        (["Đối tượng thực hiện"], f"Ai có thể thực hiện thủ tục {display_name}?"),
        (["Căn cứ pháp lý", "Cơ quan ban hành", "Cơ quan phối hợp"], f"Thủ tục {display_name} được quy định dựa trên văn bản pháp luật nào?"),
    ]

    primary_suggestions = [q for f_list, q in primary_pool if not is_asked(f_list)]
    secondary_suggestions = [q for f_list, q in secondary_pool if not is_asked(f_list)]

    results = []
    if primary_suggestions:
        results.extend(random.sample(primary_suggestions, min(2, len(primary_suggestions))))
        
    if secondary_suggestions:
        results.extend(random.sample(secondary_suggestions, min(3 - len(results), len(secondary_suggestions))))
        
    random.shuffle(results)
    return results[:3]


def _rag_response(
    answer: str,
    procedure_name: str = "",
    field=None,
    docs=None,
    candidates=None,
    suggested_questions=None,
    show_metadata: bool = True,
):
    procedure_id = _get_proc_id_by_name(procedure_name, docs=docs) if procedure_name else ""

    selected_procedure = None
    if procedure_name and show_metadata:
        selected_procedure = {
            "id": procedure_id,
            "name": procedure_name,
            "display_name": _clean_procedure_display_name(procedure_name),
        }

    if suggested_questions is None:
        suggested_questions = build_suggested_questions(procedure_name, field)

    sources = _build_sources(docs or []) if show_metadata else []

    return {
        "answer": answer,
        "suggested_questions": suggested_questions,
        "selected_procedure": selected_procedure,
        "sources": sources,
        "procedure_candidates": candidates or [],
        "show_metadata": bool(show_metadata and (selected_procedure or sources)),
    }


def _save_selected_context(session_id: str, procedure_name: str, docs=None, score=None, field=None):
    if not procedure_name or not save_selected_procedure:
        return

    procedure_id = _get_proc_id_by_name(procedure_name, docs=docs)
    save_selected_procedure(
        session_id=session_id,
        procedure_id=procedure_id,
        procedure_name=procedure_name,
        score=score,
        field=field,
    )


def _get_session_selected(session_id: str):
    if not get_selected_procedure:
        return None

    try:
        return get_selected_procedure(session_id)
    except Exception as e:
        print(f"[SESSION CONTEXT WARNING]: {e}")
        return None


def _get_procedure_context_docs(db, procedure_name: str, query: str = "", limit: int = 8, procedure_id: str = ""):
    """
    Lấy parent context Markdown của thủ tục đã chọn.
    Ưu tiên procedure_id/id để khóa nguồn, sau đó mới fallback theo tên thủ tục.
    """
    if not procedure_name and not procedure_id:
        return []

    docs = _scroll_docs_by_procedure(
        db=db,
        procedure_name=procedure_name,
        procedure_id=procedure_id,
        limit=limit,
        section_type="procedure_full",
    )

    if docs:
        return _sort_docs_natural(docs)

    # Fallback khi chưa reload vector DB bằng chunker mới.
    docs = _scroll_docs_by_procedure(
        db=db,
        procedure_name=procedure_name,
        procedure_id=procedure_id,
        limit=MAX_CONTEXT_CHUNKS,
    )

    if not docs and query:
        docs = _similarity_search_by_procedure(
            db=db,
            query=query,
            k=MAX_CONTEXT_CHUNKS,
            procedure_name=procedure_name,
            procedure_id=procedure_id,
        )

    docs = _guard_docs_by_locked_procedure(
        docs,
        procedure_name=procedure_name,
        procedure_id=procedure_id,
        stage="procedure_context_final",
    )
    return _sort_docs_natural(docs)


def _procedure_candidates_from_scores(proc_scores, limit: int = 5):
    candidates = []
    for name, score in sorted(proc_scores.items(), key=lambda item: item[1], reverse=True)[:limit]:
        candidates.append({
            "id": _get_proc_id_by_name(name),
            "name": name,
            "score": float(score) if isinstance(score, (int, float)) else score,
        })
    return candidates


def ask_rag(db, query, session_id, history=None, procedure_id: str = "", procedure_name: str = ""):
    request_started_at = time.perf_counter()
    history = history or []

    if db is None:
        return _rag_response("Hệ thống cơ sở dữ liệu hiện không khả dụng. Vui lòng thử lại sau.")

    raw_query = (query or "").strip()
    if not raw_query:
        return _rag_response("Vui lòng nhập câu hỏi.")

    # Request lock: frontend/API có thể truyền thủ tục hiện tại khi người dùng đang ở trang chi tiết.
    # Giữ tương thích ngược: nếu không truyền thì pipeline chạy theo session/detect như cũ.
    request_proc_id, request_proc_name = _resolve_locked_procedure(
        procedure_id=procedure_id,
        procedure_name=procedure_name,
    )
    request_lock_active = bool(request_proc_id or request_proc_name)

    selected_context = _get_session_selected(session_id)
    selected_proc_name = (selected_context or {}).get("name") or ""
    selected_proc_id = _normalize_procedure_id(
        (selected_context or {}).get("procedure_id")
        or (selected_context or {}).get("id")
        or _get_proc_id_by_name(selected_proc_name)
    )

    if request_proc_id or request_proc_name:
        print(
            f"[REQUEST LOCK ACTIVE] request_proc_id={request_proc_id or None} "
            f"request_proc_name={request_proc_name or None}"
        )

    # 0. Người dùng chỉ xác nhận đã hiểu: không kéo ngữ cảnh cũ ra trả lời tiếp.
    if _is_acknowledgement_query(raw_query):
        return _rag_response(
            _build_acknowledgement_answer(),
            suggested_questions=[],
            show_metadata=False,
        )

    # 1. Xã giao thật sự: không gợi ý, không metadata, không RAG.
    intent_answer = handle_intent(raw_query)
    if intent_answer and _is_social_intent_query(raw_query):
        return _rag_response(
            intent_answer,
            suggested_questions=[],
            show_metadata=False,
        )

    # 2. Lịch sử hội thoại
    if history and len(history) > 0:
        history_formatted = "LỊCH SỬ HỘI THOẠI:\n" + "\n".join([
            f"{'Người dùng' if h['role'] == 'user' else 'Chuyên viên'}: {h['content']}"
            for h in history[-HISTORY_MESSAGES_FOR_REWRITE:]
        ])
    else:
        history_formatted = "LỊCH SỬ HỘI THOẠI:\n(Chưa có)"

    # 3. Nhận diện thủ tục và field trên câu hiện tại.
    # Nếu request đã truyền procedure_id/procedure_name thì khóa luôn thủ tục đó,
    # không để resolver/vector search kéo sang thủ tục khác.
    session_lock_active = False

    if request_lock_active:
        explicit_proc = request_proc_name or _resolve_procedure_name_by_id(request_proc_id)
        explicit_proc_id = request_proc_id or _get_proc_id_by_name(explicit_proc)
    else:
        # Vẫn cho resolver đọc câu hiện tại, nhưng selected_procedure của session
        # được quyền thắng nếu câu này là follow-up hoặc resolver chỉ đoán yếu.
        explicit_proc = detect_procedure_name(raw_query, history=None, raw_query=raw_query)
        explicit_proc_id = _get_proc_id_by_name(explicit_proc)

        field_probe = detect_field(raw_query, procedure_name=selected_proc_name or explicit_proc)
        if _should_keep_selected_context(
            raw_query=raw_query,
            selected_proc_name=selected_proc_name,
            explicit_proc=explicit_proc,
            field=field_probe,
        ):
            session_lock_active = True
            explicit_proc = selected_proc_name
            explicit_proc_id = selected_proc_id or _get_proc_id_by_name(selected_proc_name)
            print(
                f"[SESSION LOCK ACTIVE] session_proc_id={explicit_proc_id or None} "
                f"session_proc_name={explicit_proc or None}"
            )

    field_before_rewrite = detect_field(raw_query, procedure_name=explicit_proc or selected_proc_name)

    route_explicit_proc = explicit_proc or ("__LOCKED_BY_PROCEDURE_ID__" if request_proc_id else "")
    query_route = _classify_conversation_route(
        raw_query=raw_query,
        selected_proc_name=selected_proc_name,
        explicit_proc=route_explicit_proc,
        field=field_before_rewrite,
    )
    print(
        f"[QUERY ROUTE]: route={query_route} selected={selected_proc_name or None} "
        f"selected_id={selected_proc_id or None} explicit={explicit_proc or None} "
        f"explicit_id={explicit_proc_id or None} field={field_before_rewrite}"
    )

    # 3.0. Câu hỏi dạng danh sách/đếm nhóm thủ tục.
    # Trả lời bằng list từ JSON local, không chọn 1 thủ tục đơn lẻ, không gọi Qdrant/LLM.
    if query_route == ROUTE_LIST_QUERY:
        list_items, total_count = _find_related_procedures_for_list(
            raw_query=raw_query,
            selected_proc_name=selected_proc_name,
            limit=12,
        )
        answer = _format_related_procedures_answer(raw_query, list_items, total_count)
        candidates = [
            {
                "id": str(item.get("id", "")),
                "name": item.get("name") or (item.get("content") or {}).get("Tên thủ tục") or "",
                "score": None,
            }
            for item in list_items
        ]
        return _rag_response(
            answer,
            candidates=candidates,
            suggested_questions=[],
            show_metadata=False,
        )

    # 3.1. Câu không có bằng chứng thủ tục hành chính và không phải nối tiếp hợp lệ.
    # Dừng sớm để không tốn Qdrant/LLM và không sinh gợi ý sai.
    if query_route == ROUTE_OUT_OF_DOMAIN:
        return _rag_response(
            _build_out_of_domain_answer(),
            suggested_questions=[],
            show_metadata=False,
        )

    # 3.2. Có vẻ liên quan hành chính nhưng chưa chắc thủ tục nào.
    # Độ chính xác ưu tiên hơn đoán mò, nên hỏi lại nhẹ nhàng.
    if query_route == ROUTE_ASK_CLARIFY:
        return _rag_response(
            _build_clarify_answer(selected_proc_name=selected_proc_name),
            suggested_questions=[],
            show_metadata=False,
        )

    use_selected_context = False
    detected_proc_before_rewrite = explicit_proc
    detected_proc_id_before_rewrite = explicit_proc_id

    # 3.3. Câu nối tiếp hợp lệ: giữ thủ tục đang chọn, không gọi history/Qdrant toàn DB.
    if query_route == ROUTE_CONTINUE_CONTEXT:
        detected_proc_before_rewrite = selected_proc_name
        detected_proc_id_before_rewrite = selected_proc_id
        use_selected_context = True
        print(f"[SESSION CONTEXT HIT]: proc={selected_proc_name} proc_id={selected_proc_id or None} field={field_before_rewrite}")

    # 3.4. Câu nêu thủ tục mới/đang đổi thủ tục thì dùng thủ tục trên câu hiện tại.
    if query_route == ROUTE_SWITCH_PROCEDURE:
        # CHỐT CHẶN: Chỉ switch nếu thủ tục explicit mới có điểm cao hơn đáng kể
        # hoặc không phải là các từ khóa nhiễu từ câu trước đó.
        # Nếu frontend đã truyền request lock thì tin request lock, không chặn bằng heuristic cũ.
        is_nhiu = any(kw in (explicit_proc or "").lower() for kw in ["hợp quy", "tiêu chuẩn"])
        if is_nhiu and selected_proc_name and not (request_proc_id or request_proc_name):
            print(f"[SECURITY BLOCK]: Chặn switch nhầm sang {explicit_proc}")
            detected_proc_before_rewrite = selected_proc_name
            detected_proc_id_before_rewrite = selected_proc_id
        else:
            use_selected_context = False
            detected_proc_before_rewrite = explicit_proc
            detected_proc_id_before_rewrite = explicit_proc_id

    # 3.5. Chỉ fallback sang history khi câu hiện tại đã có bằng chứng hành chính,
    # không phải ngoài phạm vi và không phải câu hỏi mơ hồ.
    if (
        not detected_proc_before_rewrite
        and query_route in {ROUTE_ADMIN_CONFIDENT, ROUTE_SWITCH_PROCEDURE}
        and not _is_context_switch_query(raw_query)
    ):
        detected_proc_before_rewrite = detect_procedure_name(
            raw_query,
            history=history,
            raw_query=raw_query,
        )
        detected_proc_id_before_rewrite = _get_proc_id_by_name(detected_proc_before_rewrite)

    # 3.6. Câu điều hướng: "tôi chưa biết làm thế nào", "hướng dẫn tôi..."
    # Nếu có thủ tục đang chọn hoặc câu đã nêu thủ tục, trả lời hướng dẫn ngay, không RAG.
    if _is_guidance_query(raw_query):
        guide_proc = detected_proc_before_rewrite or selected_proc_name

        if guide_proc:
            _save_selected_context(session_id, guide_proc, docs=[], score=1.0, field=None)

        answer = _build_guidance_answer(guide_proc)
        return _rag_response(
            answer,
            procedure_name=guide_proc,
            field=None,
            docs=[],
            candidates=[],
            suggested_questions=(
                [
                    "Hồ sơ cần chuẩn bị gồm những gì?",
                    "Tôi cần nộp hồ sơ ở đâu?",
                    "Bao lâu có kết quả?",
                ] if guide_proc else []
            ),
            show_metadata=False,
        )

    if not detected_proc_before_rewrite and not selected_proc_name and _is_low_information_query(raw_query):
        return _rag_response(
            _build_clarify_answer(),
            suggested_questions=[],
            show_metadata=False,
        )

    # Nếu đã có selected context, không gọi rewrite LLM nữa.
    # Câu hỏi kiểu "vậy lệ phí thì sao", "nộp ở đâu", "còn thời hạn"
    # sẽ đi thẳng vào context thủ tục đã chọn.
    should_rewrite = _should_rewrite_query(
        raw_query=raw_query,
        history=history,
        detected_proc=detected_proc_before_rewrite,
        detected_field=field_before_rewrite,
    )
    if use_selected_context or request_lock_active or session_lock_active:
        should_rewrite = False

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
            f"field={field_before_rewrite} selected_context={use_selected_context}"
        )

    if rewritten:
        query = rewritten
        print(f"[REWRITE]: {query}")
        # Nếu đã có request/session lock thì không để rewrite kéo sang thủ tục khác.
        if request_lock_active:
            detected_proc = request_proc_name or detected_proc_before_rewrite or _resolve_procedure_name_by_id(request_proc_id)
        elif session_lock_active:
            detected_proc = selected_proc_name or detected_proc_before_rewrite
        else:
            detected_proc = detected_proc_before_rewrite or detect_procedure_name(query, history=history, raw_query=query)
    else:
        query = raw_query
        if request_lock_active:
            detected_proc = request_proc_name or detected_proc_before_rewrite or _resolve_procedure_name_by_id(request_proc_id)
        elif session_lock_active:
            detected_proc = selected_proc_name or detected_proc_before_rewrite
        else:
            detected_proc = detected_proc_before_rewrite or detect_procedure_name(query, history=history, raw_query=raw_query)

    detected_proc_id = (
        request_proc_id
        or detected_proc_id_before_rewrite
        or _get_proc_id_by_name(detected_proc)
    )

    # Nếu chỉ có mã mà chưa có tên, cố gắng lấy tên để hiển thị/gợi ý.
    if detected_proc_id and not detected_proc:
        detected_proc = _resolve_procedure_name_by_id(detected_proc_id)

    # Chốt lần cuối: request lock luôn thắng resolver/session memory.
    if request_lock_active:
        detected_proc_id = request_proc_id or detected_proc_id
        detected_proc = request_proc_name or _resolve_procedure_name_by_id(detected_proc_id) or detected_proc
        print(
            f"[REQUEST LOCK FINAL] locked_proc_id={detected_proc_id or None} "
            f"locked_proc_name={detected_proc or None}"
        )

    # Chốt lần cuối cho ngữ cảnh phiên: nếu đây là câu follow-up thì selected_procedure
    # thắng resolver, tránh các lỗi nhảy sang bảo hiểm, đất đai, thiết bị y tế...
    if session_lock_active and selected_proc_name:
        detected_proc = selected_proc_name
        detected_proc_id = selected_proc_id or detected_proc_id or _get_proc_id_by_name(selected_proc_name)
        use_selected_context = True
        print(
            f"[SESSION LOCK FINAL] locked_proc_id={detected_proc_id or None} "
            f"locked_proc_name={detected_proc or None}"
        )

    llm_query = query
    query = normalize_query(query)
    field = field_before_rewrite or detect_field(query, procedure_name=detected_proc or selected_proc_name)

    # Cache phải gắn với mã/tên thủ tục đang chọn, nếu không câu "lệ phí bao nhiêu" sẽ bị dùng nhầm giữa các session.
    query_key = f"{detected_proc_id or detected_proc or selected_proc_name or ''}::{query.lower().strip()}"
    cached_answer = _cache_get(query_key)
    if cached_answer:
        return _rag_response(cached_answer, procedure_name=detected_proc or selected_proc_name, field=field)

    try:
        print(f"\n===== XỬ LÝ TRUY VẤN: {query} =====")
        print(
            f"[DETECTED]: explicit_proc={explicit_proc} explicit_id={explicit_proc_id or None} "
            f"detected_proc={detected_proc} detected_id={detected_proc_id or None} field={field}"
        )

        docs = []
        candidates = []
        skip_rerank = False
        
        has_hard_lock = request_lock_active or session_lock_active

        # 4. Fast path: đã biết thủ tục + field thì chỉ lấy chunk field trong thủ tục đó.
        fast_exact_docs = _exact_field_retrieval(
            db=db,
            query=query,
            detected_proc=detected_proc,
            detected_proc_id=detected_proc_id,
            field=field,
        )

        parent_fallback_docs = []
        if detected_proc and field and "Yêu cầu điều kiện" in _as_field_list(field) and not fast_exact_docs:
            # Field điều kiện trong dữ liệu DVC thường rỗng/"Không có thông tin".
            # Theo thiết kế gốc: đã biết thủ tục thì kéo parent context toàn thủ tục để đọc sâu trong chính thủ tục đó,
            # không dừng sớm ở một field rỗng và không nhảy sang thủ tục khác.
            parent_fallback_docs = _get_procedure_context_docs(
                db,
                detected_proc,
                query=query,
                limit=MAX_CONTEXT_CHUNKS,
                procedure_id=detected_proc_id,
            )

        if fast_exact_docs:
            docs = fast_exact_docs
            skip_rerank = True
            print("[RETRIEVAL MODE]: exact field by selected/detected procedure")

        elif parent_fallback_docs:
            docs = parent_fallback_docs
            skip_rerank = True
            print("[RETRIEVAL MODE]: condition field empty -> parent procedure context")

        # 4.1. Nếu câu hỏi mơ hồ nhưng đã có thủ tục chính, lấy parent context của thủ tục đó.
        elif detected_proc and (use_selected_context or selected_proc_name == detected_proc):
            docs = _get_procedure_context_docs(
                db,
                detected_proc,
                query=query,
                procedure_id=detected_proc_id,
            )
            skip_rerank = True
            print("[RETRIEVAL MODE]: selected procedure parent context")

        else:
            # 5. Retrieval khi chưa đủ context.
            if has_hard_lock:
                print(f"[SYSTEM]: Khóa truy hồi theo thủ tục (HARD LOCK): {detected_proc} | id={detected_proc_id or None}")
                docs_filter = _similarity_search_by_procedure(
                    db=db,
                    query=query,
                    k=RETRIEVAL_SEMANTIC_K,
                    procedure_name=detected_proc,
                    procedure_id=detected_proc_id,
                )
                if not docs_filter:
                    docs_filter = _scroll_docs_by_procedure(
                        db=db,
                        procedure_name=detected_proc,
                        procedure_id=detected_proc_id,
                        limit=EXACT_PROC_SCROLL_LIMIT,
                    )
                docs.extend(docs_filter)
            elif detected_proc or detected_proc_id:
                print(f"[SYSTEM]: Khóa truy hồi theo thủ tục (SOFT LOCK): {detected_proc} | id={detected_proc_id or None}")
                docs_filter = _similarity_search_by_procedure(
                    db=db,
                    query=query,
                    k=max(FILTER_SEARCH_K, RETRIEVAL_SEMANTIC_K),
                    procedure_name=detected_proc,
                    procedure_id=detected_proc_id,
                )
                docs.extend(docs_filter)
            else:
                retriever_semantic = db.as_retriever(search_kwargs={"k": RETRIEVAL_SEMANTIC_K})
                docs_semantic = retriever_semantic.invoke(query)
                docs.extend(docs_semantic)

        # 6. Deduplicate
        unique_docs = []
        seen_content = set()
        for d in docs:
            if d.page_content not in seen_content:
                unique_docs.append(d)
                seen_content.add(d.page_content)
        docs = unique_docs

        if has_hard_lock:
            docs = _guard_docs_by_locked_procedure(
                docs,
                procedure_name=detected_proc,
                procedure_id=detected_proc_id,
                stage="post_dedup_hard_lock",
            )
        elif detected_proc or detected_proc_id:
            docs = _guard_docs_by_locked_procedure(
                docs,
                procedure_name=detected_proc,
                procedure_id=detected_proc_id,
                stage="post_dedup",
            )

        if has_hard_lock and not docs:
            answer = f"Hiện dữ liệu chưa tìm thấy thông tin phù hợp trong thủ tục {detected_proc or detected_proc_id}."
            if detected_proc:
                _save_selected_context(session_id, detected_proc, docs=[], score=1.0, field=field)
            return _rag_response(answer, procedure_name=detected_proc or selected_proc_name, field=field)

        if not docs:
            answer = "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."
            if detected_proc:
                _save_selected_context(session_id, detected_proc, docs=[], score=1.0, field=field)
            return _rag_response(answer, procedure_name=detected_proc or selected_proc_name, field=field)

        # CHỐT CHẶN TỐI THƯỢNG: Chỉ lấy tên từ metadata NẾU VÀ CHỈ NẾU doc đó khớp với ID đã bị khóa cứng.
        # Tuyệt đối không lấy bừa từ chunk top 1 nếu đang search tự do (cả ID và Tên đều đang trống).
        if not detected_proc and detected_proc_id:
            for doc in docs:
                meta = getattr(doc, "metadata", {}) or {}
                if _same_procedure_id(detected_proc_id, _get_metadata_procedure_id(meta)):
                    detected_proc = _get_metadata_procedure_name(meta)
                    if detected_proc:
                        break

        # 7. Rerank nếu cần
        if skip_rerank:
            for d in docs:
                d.metadata['score'] = 1.0
            print("[RERANK SKIP]: dùng context đã chọn hoặc exact retrieval")
        elif reranker:
            rerank_started = time.perf_counter()
            pairs = [(llm_query, d.page_content) for d in docs]
            scores = reranker.predict(pairs)
            rerank_ms = int((time.perf_counter() - rerank_started) * 1000)
            print(f"[RERANK DONE]: docs={len(docs)} latency_ms={rerank_ms}")
            for i, d in enumerate(docs):
                d.metadata["score"] = float(scores[i])
            docs = sorted(docs, key=lambda x: x.metadata['score'], reverse=True)
        else:
            for d in docs:
                d.metadata['score'] = 1.0

        # 8. Chọn thủ tục thắng cuộc
        proc_scores = {}
        for d in docs[:8]:
            p_name = d.metadata.get("name")
            if p_name:
                proc_scores[p_name] = max(proc_scores.get(p_name, d.metadata['score']), d.metadata['score'])

        candidates = _procedure_candidates_from_scores(proc_scores, limit=5)

        if detected_proc:
            allowed_proc_names = [detected_proc]
            winner_score = proc_scores.get(detected_proc, 1.0)
            print(f"[WINNER - DETECTED/SELECTED PROC]: {allowed_proc_names}")
        else:
            top_procs = sorted(proc_scores.items(), key=lambda item: item[1], reverse=True)[:2]
            allowed_proc_names = [p[0] for p in top_procs] if top_procs else ["Thủ tục không xác định"]
            detected_proc = allowed_proc_names[0] if allowed_proc_names[0] != "Thủ tục không xác định" else None
            winner_score = top_procs[0][1] if top_procs else None
            
            # Cập nhật luôn detected_proc_id cho thủ tục vừa chiến thắng để khóa Guardrail Bước 10
            if detected_proc and not detected_proc_id:
                detected_proc_id = _get_proc_id_by_name(detected_proc)
                
            print(f"[WINNER - BLIND SEARCH]: {allowed_proc_names}")

        # 9. Parent-child bổ sung method nếu câu hỏi liên quan thời hạn/phí/cách thức.
        seen_content = {d.page_content for d in docs}
        if (detected_proc or detected_proc_id) and _is_method_related_field(field) and not skip_rerank:
            extra_docs = _similarity_search_by_procedure(
                db=db,
                query=query,
                k=PARENT_METHOD_K,
                procedure_name=detected_proc,
                procedure_id=detected_proc_id,
                section_type="method",
            )
            for ed in extra_docs:
                if ed.page_content not in seen_content:
                    ed.metadata['score'] = 0.99
                    docs.insert(0, ed)
                    seen_content.add(ed.page_content)
        else:
            print(f"[PARENT METHOD SKIP]: field={field}")

        # 10. Lọc chunk theo winner và field.
        final_docs = _guard_docs_by_locked_procedure(
            docs,
            procedure_name=detected_proc,
            procedure_id=detected_proc_id,
            stage="before_field_filter",
        )
        if field:
            field_docs = [d for d in final_docs if d.metadata.get("field") in field]
            if field_docs:
                other_docs = [d for d in final_docs if d not in field_docs]
                final_docs = field_docs + other_docs
                print(f"[FIELD FILTER]: Đã ưu tiên các mục {field} lên đầu")

        # 10.1 Nếu field rõ mà final_docs chưa đủ, lấy thêm exact field bằng scroll.
        if (detected_proc or detected_proc_id) and field and not skip_rerank:
            exact_field_docs = _exact_field_retrieval(
                db,
                query,
                detected_proc,
                field,
                detected_proc_id=detected_proc_id,
            )
            if exact_field_docs:
                if STRICT_FIELD_CONTEXT:
                    final_docs = _sort_docs_natural(exact_field_docs)
                    print(f"[STRICT FIELD CONTEXT]: chỉ giữ context theo field {field}")
                else:
                    final_docs = _sort_docs_natural(exact_field_docs + final_docs)

        final_docs = _guard_docs_by_locked_procedure(
            final_docs,
            procedure_name=detected_proc,
            procedure_id=detected_proc_id,
            stage="final_docs",
        )
        final_docs = _sort_docs_natural(final_docs)

        if has_hard_lock and not final_docs:
            answer = f"Hiện dữ liệu chưa có thông tin này trong thủ tục {detected_proc or detected_proc_id}."
            if detected_proc:
                _save_selected_context(session_id, detected_proc, docs=[], score=1.0, field=field)
            return _rag_response(answer, procedure_name=detected_proc or selected_proc_name, field=field)

        if not final_docs:
            answer = (
                "Hiện dữ liệu của thủ tục này chưa ghi thông tin phù hợp để trả lời câu hỏi. "
                "Bạn có thể hỏi lại rõ hơn theo phần hồ sơ, nơi nộp, thời hạn hoặc phí/lệ phí."
            )
            if detected_proc:
                _save_selected_context(session_id, detected_proc, docs=[], score=1.0, field=field)
            return _rag_response(answer, procedure_name=detected_proc or selected_proc_name, field=field)

        seen = set()
        context_chunks = []
        context_docs = []
        for d in final_docs:
            if d.page_content in seen:
                continue
            context_chunks.append(d.page_content)
            context_docs.append(d)
            seen.add(d.page_content)
            if len(context_chunks) >= MAX_CONTEXT_CHUNKS:
                break

        context = "\n\n".join(context_chunks)
        print(
            f"[CONTEXT] chunks={len(context_chunks)} chars={len(context)} "
            f"tokens≈{estimate_tokens(context)}"
        )

        if detected_proc:
            _save_selected_context(session_id, detected_proc, docs=context_docs, score=winner_score, field=field)

        # 11. Fast direct answer cho bẫy đơn giản.
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
                print(f"[RAG DIRECT TRAP OK] session_id={session_id} latency_ms={elapsed_ms}")
                return _rag_response(direct_trap_answer, detected_proc, field, context_docs, candidates)

        # 12. Direct answer cho câu rõ field, kể cả câu hỏi tiếp theo dạng "thế lệ phí thì sao".
        if ENABLE_DIRECT_ANSWER and detected_proc and field and not _is_trap_like(raw_query):
            direct_answer = _build_direct_answer(detected_proc, field, context_chunks, raw_query=raw_query)

            if direct_answer:
                _cache_set(query_key, direct_answer)
                elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
                print(f"[RAG DIRECT OK] session_id={session_id} latency_ms={elapsed_ms}")
                return _rag_response(direct_answer, detected_proc, field, context_docs, candidates)

        # 13. Generate bằng LLM với context đã chọn.
        prompt = f"""{SYSTEM_PROMPT}

{history_formatted}

CONTEXT TÀI LIỆU (Cập nhật mới nhất):
{context}

CÂU HỎI HIỆN TẠI: {raw_query}

LƯU Ý TRẢ LỜI:
- Chỉ trả lời đúng phần người dùng hỏi.
- Nếu người dùng hỏi hồ sơ/giấy tờ thì không tự thêm thời hạn, lệ phí, căn cứ pháp lý.
- Nếu người dùng hỏi thời hạn/lệ phí thì không tự liệt kê toàn bộ hồ sơ.
- Nếu người dùng hỏi điều kiện, độ tuổi, quan hệ họ hàng hoặc câu hỏi dạng "có được không", hãy tìm trong toàn bộ CONTEXT của chính thủ tục đó.
- Nếu CONTEXT chỉ ghi "đủ điều kiện theo luật" nhưng không ghi chi tiết điều kiện, phải nói rõ dữ liệu thủ tục chưa ghi chi tiết, không được tự suy ra số tuổi hay phạm vi họ hàng.
- Không tóm tắt toàn bộ thủ tục khi người dùng hỏi điều kiện; chỉ trả lời điều kiện hoặc nói thiếu dữ liệu điều kiện.
- Nếu câu hỏi có giả định sai, hãy đính chính ngắn gọn rồi nêu thông tin đúng theo CONTEXT.
- Nếu đang có thủ tục chính trong phiên, ưu tiên trả lời trong phạm vi thủ tục đó, trừ khi người dùng nêu rõ thủ tục khác.
- Phải ưu tiên thông tin trong CONTEXT trước.
- ĐỐI VỚI ĐIỀU KIỆN KẾT HÔN/ĐẤT ĐAI: Nếu CONTEXT ghi "quy định pháp luật", bạn ĐƯỢC PHÉP suy luận dựa trên luật hiện hành (Nam từ 20, Nữ từ 18, cấm kết hôn 3 đời, không hôn nhân đồng giới...) để giải đáp cho người dùng, nhưng phải bắt đầu bằng câu "Theo quy định pháp luật...".
- KHÔNG TỰ BỊA SỐ LIỆU: Nếu hỏi về mốc ngày, phí, nếu CONTEXT Lai Châu không có, hãy báo rõ địa phương chưa ghi nhận, không tự lấy dữ liệu tỉnh khác.
- Phân biệt rõ các khoảng thời gian không tính vào hạn giải quyết (xác minh nghĩa vụ tài chính, niêm yết công khai, đăng tin...).
TRẢ LỜI:"""

        answer = smart_llm_invoke(prompt)
        if answer:
            answer = answer.replace('**', '')
            answer = re.sub(r'(?m)^\s*\*\s+', '- ', answer)
            answer = re.sub(r'\n{3,}', '\n\n', answer).strip()
            _cache_set(query_key, answer)
            elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
            print(
                f"[RAG OK] session_id={session_id} latency_ms={elapsed_ms} "
                f"query_tokens≈{estimate_tokens(raw_query)} answer_tokens≈{estimate_tokens(answer)}"
            )
            return _rag_response(answer, detected_proc or selected_proc_name, field, context_docs, candidates)

        return _rag_response(
            "Hiện tại dịch vụ AI chưa phản hồi. Vui lòng thử lại sau.",
            procedure_name=detected_proc or selected_proc_name,
            field=field,
            docs=context_docs,
            candidates=candidates,
        )

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
        print(f"[CRITICAL ERROR] latency_ms={elapsed_ms}: {e}")
        return _rag_response(
            "Hệ thống đang bận, vui lòng thử lại sau.",
            procedure_name=detected_proc or selected_proc_name,
            field=field,
        )