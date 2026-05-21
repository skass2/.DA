import importlib.util
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

# optional fallback OpenRouter nếu có cài
if importlib.util.find_spec("langchain_openai"):
    from langchain_openai import ChatOpenAI
else:
    ChatOpenAI = None


PRIMARY_MODEL = os.getenv("CHATBOT_PRIMARY_MODEL", "gemini-2.5-flash")
LIGHTWEIGHT_MODEL = os.getenv("CHATBOT_LIGHTWEIGHT_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("CHATBOT_TIMEOUT", "30"))

_llm = None
_lightweight_llm = None
_fallback_llms = None


def _build_gemini(model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
    )


def get_llm():
    """
    Primary LLM: Gemini 2.5 Flash với temperature thấp để ưu tiên độ chính xác.
    """
    global _llm

    if _llm is None:
        try:
            _llm = _build_gemini(PRIMARY_MODEL)
            print(f"[LLM] Using Primary: {PRIMARY_MODEL} (temperature={LLM_TEMPERATURE})")
        except Exception as e:
            print("[LLM ERROR] Gemini primary init failed:", e)
            _llm = None

    return _llm


def get_lightweight_llm():
    """
    Lightweight LLM: Gemini 2.5 Flash Lite cho rewrite/fallback và request nhẹ.
    """
    global _lightweight_llm

    if _lightweight_llm is None:
        try:
            _lightweight_llm = _build_gemini(LIGHTWEIGHT_MODEL)
            print(f"[LLM] Using Lightweight: {LIGHTWEIGHT_MODEL} (temperature={LLM_TEMPERATURE})")
        except Exception as e:
            print("[LLM ERROR] Gemini lightweight init failed:", e)
            _lightweight_llm = None

    return _lightweight_llm


def get_fallback_llms():
    """
    Fallback chain:
    Gemini 2.5 Flash Lite
    → Qwen 2.5:7b Ollama
    → Llama 3.2 Ollama
    → OpenRouter
    """
    global _fallback_llms

    if _fallback_llms is not None:
        return _fallback_llms

    llms = []

    lightweight = get_lightweight_llm()
    if lightweight is not None:
        llms.append(lightweight)
        print(f"[Fallback] Added {LIGHTWEIGHT_MODEL}")

    # Qwen 2.5:7b Ollama Local
    try:
        llms.append(ChatOllama(
            model="qwen2.5:7b",
            temperature=LLM_TEMPERATURE,
        ))
        print("[Fallback] Added Ollama qwen2.5:7b")
    except Exception as e:
        print("[Fallback] Ollama qwen2.5:7b not available:", e)

    # Llama 3.2 Ollama Local
    try:
        llms.append(ChatOllama(
            model="llama3.2",
            temperature=LLM_TEMPERATURE,
        ))
        print("[Fallback] Added Ollama llama3.2")
    except Exception as e:
        print("[Fallback] Ollama llama3.2 not available:", e)

    # OpenRouter nếu có API key
    if ChatOpenAI:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if api_key:
            try:
                llms.append(ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct"),
                    temperature=LLM_TEMPERATURE,
                    timeout=LLM_TIMEOUT,
                ))
                print("[Fallback] Added OpenRouter")
            except Exception as e:
                print("[Fallback ERROR] OpenRouter:", e)

    _fallback_llms = llms
    return llms