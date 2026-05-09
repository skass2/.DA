import json
import random
from rapidfuzz import fuzz

try:
    with open("data/intents.json", "r", encoding="utf-8") as f:
        INTENTS = json.load(f)["intents"]
    INTENTS = sorted(INTENTS, key=lambda x: x.get("priority", 99))
except:
    INTENTS = []

def handle_intent(query: str):
    q = query.lower().strip()
    # Nếu câu dài > 7 từ thì không phải là câu xã giao chào hỏi thông thường
    if len(q.split()) > 7: return None

    for intent in INTENTS:
        for ex in intent["user_examples"]:
            if fuzz.ratio(ex.lower(), q) > 85 or (len(q) < 10 and ex.lower() in q):
                res = random.choice(intent["bot_responses"])
                if intent.get("follow_up"):
                    res += "\n\n" + random.choice(intent["follow_up"])
                return res
    return None