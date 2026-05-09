import json

try:
    with open("data/synonyms.json", encoding="utf-8") as f:
        SYNONYMS = json.load(f)
except:
    SYNONYMS = {}

def normalize_query(query: str):
    q = " " + query.lower() + " "
    for key, values in SYNONYMS.items():
        for v in values:
            target = " " + v.lower() + " "
            if target in q:
                q = q.replace(target, f" {key} ")
    return q.strip()