import json

def load_data(path="data/procedures.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
