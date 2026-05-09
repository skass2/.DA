import json
import re
import unicodedata
from typing import List, Dict, Set

# ===== CONFIG =====
STOPWORDS = {
    "và", "của", "cho", "về", "trong", "theo", "tại", "được", "với",
    "các", "những", "khi", "đó", "này", "là", "có", "sự", "việc", 
    "để", "qua", "từ", "lên", "xuống", "ra", "vào", "một", "những"
}

MIN_WORD_LEN = 2


# ===== NORMALIZE =====
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ===== REMOVE ACCENTS =====
def remove_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


# ===== EXTRACT NGRAM =====
def generate_ngrams(words: List[str], n=2) -> List[str]:
    return [
        " ".join(words[i:i+n])
        for i in range(len(words) - n + 1)
    ]


# ===== CLEAN WORDS =====
def clean_words(words: List[str]) -> List[str]:
    return [
        w for w in words
        if w not in STOPWORDS and len(w) >= MIN_WORD_LEN
    ]


# ===== GENERATE KEYWORDS =====
def generate_keywords(text: str) -> Set[str]:
    text = normalize_text(text)
    words = text.split()

    words = clean_words(words)

    # BỎ TỪ ĐƠN (1 âm tiết) vì tiếng Việt 1 âm tiết thường vô nghĩa và gây nhiễu
    keywords = set()

    # Chỉ lấy các cụm từ (n-gram) từ 2 đến 5 âm tiết
    keywords.update(generate_ngrams(words, 2))
    keywords.update(generate_ngrams(words, 3))
    keywords.update(generate_ngrams(words, 4))
    keywords.update(generate_ngrams(words, 5))

    return keywords


# ===== GENERATE USER QUERIES =====
def generate_user_queries(name: str) -> List[str]:
    base = normalize_text(name)

    return [
        f"làm {base}",
        f"thủ tục {base}",
        f"cách {base}",
        f"hướng dẫn {base}",
        f"điều kiện {base}",
    ]


# ===== MAIN BUILD =====
def build_entities(procedures: List[Dict]) -> Dict[str, List[str]]:
    result = {}
    for proc in procedures:
        name = proc.get("name", "")
        content = proc.get("content", {})

        all_keywords = set()

        # 1. CHỈ lấy từ khóa từ Tên thủ tục (Đây là mỏ neo quan trọng nhất)
        all_keywords.update(generate_keywords(name))

        # 2. CHỈ lấy thêm từ 'Lĩnh vực' (Ví dụ: "Ngân hàng", "Đăng kiểm") 
        # để tăng độ chính xác khi hỏi theo nhóm
        field = content.get("Lĩnh vực", "")
        if field:
            all_keywords.add(normalize_text(field))

        # 3. Thêm query tự nhiên dựa trên Tên (Giữ nguyên phần này của bạn)
        all_keywords.update(generate_user_queries(name))

        # 4. THÊM các biến thể từ khóa viết tắt hoặc đặc trưng (Manual nếu cần)
        # Ví dụ: "tiết kiệm", "gửi tiền"
        
        # 5. Bản không dấu
        no_accent = {remove_accents(k) for k in all_keywords}
        all_keywords.update(no_accent)

        # LOẠI BỎ TỪ KHÓA RÁC (Rất quan trọng)
        trash_keywords = {
            "hình thức", "khác", "thực hiện", "thủ tục", "quy định", "của", "các",
            "cấp giấy", "giấy chứng", "chứng nhận", "cách làm", "hướng dẫn", "điều kiện"
        }
        # Chỉ giữ lại các từ khóa có độ dài từ 2 âm tiết trở lên
        final_keywords = {k for k in all_keywords if k not in trash_keywords and len(k.split()) >= 2}

        result[name] = sorted(final_keywords)
    return result


# ===== RUN =====
if __name__ == "__main__":
    with open("procedures.json", "r", encoding="utf-8") as f:
        procedures = json.load(f)

    entities = build_entities(procedures)

    with open("entities.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    print("✅ Build entities hoàn tất!")