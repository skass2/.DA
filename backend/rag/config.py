import importlib.util
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================================================
# LOAD ENV
# =========================================================
# File này nằm ở backend/rag/config.py
# => parents[1] là thư mục backend
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


# Optional fallback OpenRouter nếu có cài langchain_openai
if importlib.util.find_spec("langchain_openai"):
    from langchain_openai import ChatOpenAI
else:
    ChatOpenAI = None


# =========================================================
# MODEL CONFIG
# =========================================================

PRIMARY_MODEL = os.getenv("CHATBOT_PRIMARY_MODEL", "gemini-2.5-flash")
LIGHTWEIGHT_MODEL = os.getenv("CHATBOT_LIGHTWEIGHT_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("CHATBOT_TIMEOUT", "30"))


# =========================================================
# GEMINI API KEYS
# =========================================================
# Hỗ trợ cả 2 kiểu:
#
# Cách khuyên dùng:
# GEMINI_API_KEYS=key1,key2,key3
#
# Cách cũ vẫn hỗ trợ:
# GOOGLE_API_KEY=key1,key2,key3
#
# Nếu chỉ có 1 key:
# GOOGLE_API_KEY=key1

def _split_keys(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []

    return [
        key.strip()
        for key in raw_value.split(",")
        if key.strip()
    ]


def _load_gemini_api_keys() -> List[str]:
    keys = []

    # Ưu tiên biến mới
    keys.extend(_split_keys(os.getenv("GEMINI_API_KEYS")))

    # Hỗ trợ biến cũ GOOGLE_API_KEY, kể cả khi chứa nhiều key ngăn cách bằng dấu phẩy
    if not keys:
        keys.extend(_split_keys(os.getenv("GOOGLE_API_KEY")))

    # Hỗ trợ thêm tên GEMINI_API_KEY nếu bố từng dùng
    if not keys:
        keys.extend(_split_keys(os.getenv("GEMINI_API_KEY")))

    # Loại key trùng để tránh gọi lặp không cần thiết
    unique_keys = []
    seen = set()

    for key in keys:
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)

    return unique_keys


GEMINI_API_KEYS = _load_gemini_api_keys()


def _mask_key(api_key: Optional[str]) -> str:
    """
    Chỉ in 4 ký tự cuối để tránh lộ key trong log.
    """
    if not api_key:
        return "NO_KEY"

    if len(api_key) <= 8:
        return "****"

    return f"...{api_key[-4:]}"


# =========================================================
# GLOBAL CACHE
# =========================================================

_llm = None
_lightweight_llm = None
_fallback_llms = None


# =========================================================
# BUILD LLM
# =========================================================

def _build_gemini(model_name: str, api_key: Optional[str] = None):
    """
    Tạo Gemini model bằng API key cụ thể.
    """
    kwargs = {
        "model": model_name,
        "temperature": LLM_TEMPERATURE,
        "timeout": LLM_TIMEOUT,
        "max_retries": 0,
    }

    if api_key:
        kwargs["google_api_key"] = api_key

    return ChatGoogleGenerativeAI(**kwargs)


def _build_openrouter():
    """
    Tạo OpenRouter fallback nếu có OPENROUTER_API_KEY.
    """
    if not ChatOpenAI:
        return None

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return None

    try:
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct"),
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
        )
        print("[Fallback] Added OpenRouter")
        return llm

    except Exception as e:
        print("[Fallback ERROR] OpenRouter:", e)
        return None


# =========================================================
# PUBLIC GETTERS
# =========================================================

def get_llm():
    """
    Primary LLM.

    Dùng Gemini primary model với API key đầu tiên.
    Các key còn lại được đưa vào fallback.
    """
    global _llm

    if _llm is not None:
        return _llm

    if not GEMINI_API_KEYS:
        print(
            "[LLM ERROR] Không tìm thấy Gemini API key. "
            "Hãy cấu hình GEMINI_API_KEYS hoặc GOOGLE_API_KEY trong .env."
        )
        return None

    primary_key = GEMINI_API_KEYS[0]

    try:
        _llm = _build_gemini(PRIMARY_MODEL, primary_key)
        print(
            f"[LLM] Using Primary: {PRIMARY_MODEL} "
            f"(temperature={LLM_TEMPERATURE}, key#1={_mask_key(primary_key)})"
        )

    except Exception as e:
        print("[LLM ERROR] Gemini primary init failed:", e)
        _llm = None

    return _llm


def get_lightweight_llm():
    """
    Lightweight LLM.

    Dùng cho rewrite/fallback nhẹ.
    """
    global _lightweight_llm

    if _lightweight_llm is not None:
        return _lightweight_llm

    if not GEMINI_API_KEYS:
        print(
            "[LLM ERROR] Không tìm thấy Gemini API key cho lightweight model. "
            "Hãy cấu hình GEMINI_API_KEYS hoặc GOOGLE_API_KEY trong .env."
        )
        return None

    primary_key = GEMINI_API_KEYS[0]

    try:
        _lightweight_llm = _build_gemini(LIGHTWEIGHT_MODEL, primary_key)
        print(
            f"[LLM] Using Lightweight: {LIGHTWEIGHT_MODEL} "
            f"(temperature={LLM_TEMPERATURE}, key#1={_mask_key(primary_key)})"
        )

    except Exception as e:
        print("[LLM ERROR] Gemini lightweight init failed:", e)
        _lightweight_llm = None

    return _lightweight_llm


def get_fallback_llms():
    """
    Fallback chain không dùng Ollama:

    1. Gemini primary model với các key còn lại.
    2. Gemini lightweight model với tất cả key.
    3. OpenRouter nếu có cấu hình OPENROUTER_API_KEY.

    Lưu ý:
    - Nếu nhiều key cùng thuộc một Google Cloud / AI Studio project,
      việc đổi key vẫn có thể gặp chung quota project.
    - Nếu key thuộc nhiều project khác nhau, fallback sẽ hữu ích hơn.
    """
    global _fallback_llms

    if _fallback_llms is not None:
        return _fallback_llms

    llms = []

    if not GEMINI_API_KEYS:
        print("[Fallback] Không có Gemini API key để tạo fallback.")
    else:
        # 1. Primary model với các key còn lại, bỏ key#1 vì đã dùng làm primary chính
        for index, api_key in enumerate(GEMINI_API_KEYS[1:], start=2):
            try:
                llms.append(_build_gemini(PRIMARY_MODEL, api_key))
                print(
                    f"[Fallback] Added {PRIMARY_MODEL} "
                    f"key#{index}={_mask_key(api_key)}"
                )
            except Exception as e:
                print(f"[Fallback ERROR] {PRIMARY_MODEL} key#{index}:", e)

        # 2. Lightweight model với tất cả key
        for index, api_key in enumerate(GEMINI_API_KEYS, start=1):
            try:
                llms.append(_build_gemini(LIGHTWEIGHT_MODEL, api_key))
                print(
                    f"[Fallback] Added {LIGHTWEIGHT_MODEL} "
                    f"key#{index}={_mask_key(api_key)}"
                )
            except Exception as e:
                print(f"[Fallback ERROR] {LIGHTWEIGHT_MODEL} key#{index}:", e)

    # 3. OpenRouter fallback nếu có
    openrouter_llm = _build_openrouter()
    if openrouter_llm is not None:
        llms.append(openrouter_llm)

    _fallback_llms = llms
    return _fallback_llms