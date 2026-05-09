from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

# optional fallback (nếu có cài)
try:
    from langchain_openai import ChatOpenAI
except:
    ChatOpenAI = None

_llm = None
_fallback_llms = None


def get_llm():
    """
    Primary LLM: Qwen 2.5:7b (Ollama)
    """
    global _llm

    if _llm is None:
        try:
            _llm = ChatOllama(
                model="qwen2.5:7b",
                temperature=0
            )
            print("[LLM] Using Primary: Ollama (qwen2.5:7b)")
        except Exception as e:
            print("[LLM ERROR] Ollama init failed:", e)
            _llm = None
    return _llm


def get_fallback_llms():
    """
    Fallback chain: Llama 3.2 → Gemini 2.0 Flash → Gemini 2.0 Flash Lite → OpenRouter
    """
    global _fallback_llms
    
    if _fallback_llms is not None:
        return _fallback_llms
        
    llms = []

    # 1. Llama 3.2 (Ollama Local)
    try:
        llms.append(ChatOllama(
            model="llama3.2", 
            temperature=0
        ))
        print("[Fallback] Added Ollama (llama3.2)")
    except Exception as e:
        print("[Fallback] Ollama Llama 3.2 not available:", e)

    # 2. Gemini 2.0 Flash (Bản nhẹ, tốc độ cao, 15 req/min)
    try:
        llms.append(ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            timeout=30
        ))
        print("[Fallback] Added Gemini 2.0 Flash")
    except Exception as e:
        print("[Fallback ERROR] Gemini Flash:", e)

    # 3. Gemini 2.0 Flash Lite (Dự phòng hạng nhẹ tiếp theo, 15 req/min)
    try:
        llms.append(ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            timeout=30
        ))
        print("[Fallback] Added Gemini 2.0 Flash Lite")
    except Exception as e:
        print("[Fallback ERROR] Gemini 2.0 Flash Lite:", e)

    # 4. OpenRouter (nếu có)
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