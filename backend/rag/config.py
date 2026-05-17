from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

# optional fallback OpenRouter nếu có cài
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


_llm = None
_fallback_llms = None


def get_llm():
    """
    Primary LLM: Gemini 2.0 Flash
    """
    global _llm

    if _llm is None:
        try:
            _llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                timeout=30
            )
            print("[LLM] Using Primary: Gemini 2.5 Flash")
        except Exception as e:
            print("[LLM ERROR] Gemini primary init failed:", e)
            _llm = None

    return _llm


def get_fallback_llms():
    """
    Fallback chain:
    Gemini 2.5 Flash Lite
    → Gemini 2.5 Pro
    → Gemini 1.5 Pro
    → Qwen 2.5:7b Ollama
    → Llama 3.2 Ollama
    → OpenRouter
    """
    global _fallback_llms

    if _fallback_llms is not None:
        return _fallback_llms

    llms = []

    # 1. Gemini 2.0 Flash Lite
    try:
        llms.append(ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            timeout=30
        ))
        print("[Fallback] Added Gemini 2.5 Flash Lite")
    except Exception as e:
        print("[Fallback ERROR] Gemini 2.5 Flash Lite:", e)

    # 2. Gemini 1.5 Flash
    try:
        llms.append(ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            timeout=30
        ))
        print("[Fallback] Added Gemini 2.5 Pro")
    except Exception as e:
        print("[Fallback ERROR] Gemini 2.5 Pro:", e)

    # 3. Gemini 1.5 Pro
    try:
        llms.append(ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            timeout=30
        ))
        print("[Fallback] Added Gemini 1.5 Pro")
    except Exception as e:
        print("[Fallback ERROR] Gemini 1.5 Pro:", e)

    # 4. Qwen 2.5:7b Ollama Local
    try:
        llms.append(ChatOllama(
            model="qwen2.5:7b",
            temperature=0
        ))
        print("[Fallback] Added Ollama qwen2.5:7b")
    except Exception as e:
        print("[Fallback] Ollama qwen2.5:7b not available:", e)

    # 5. Llama 3.2 Ollama Local
    try:
        llms.append(ChatOllama(
            model="llama3.2",
            temperature=0
        ))
        print("[Fallback] Added Ollama llama3.2")
    except Exception as e:
        print("[Fallback] Ollama llama3.2 not available:", e)

    # 6. OpenRouter nếu có API key
    if ChatOpenAI:
        import os

        api_key = os.getenv("OPENROUTER_API_KEY")

        if api_key:
            try:
                llms.append(ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model="mistralai/mistral-7b-instruct",
                    temperature=0,
                    timeout=30
                ))
                print("[Fallback] Added OpenRouter")
            except Exception as e:
                print("[Fallback ERROR] OpenRouter:", e)

    _fallback_llms = llms
    return llms
