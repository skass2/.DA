from langchain_google_genai import ChatGoogleGenerativeAI

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            timeout=30
        )
    return _llm
def get_fallback_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        timeout=30
    )
