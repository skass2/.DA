import re
import copy
import hashlib

from typing import Any, Dict, List

from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================================================
# CONFIG
# =========================================================

# Parent/full context có thể dài hơn, child chunks giữ vừa phải để retrieval chính xác.
MAX_CHARS = 4500

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=350,
    separators=[
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
    ],
)

NOISE_PHRASES = [
    "Chọn cơ quan thực hiện",
    "Tỉnh/Thành phố",
    "Bộ ngành",
    "Phường/Xã",
    "--Chọn Phường/Xã--",
    "Đồng ý",
    "Hệ thống chỉ hiển thị những cơ quan",
    "Hệ thống chỉ hiển thị những cơ quan (Sở/xã đã áp dụng dịch vụ công)",
    "Bộ, cơ quan, địa phương cung cấp dịch vụ công trực tuyến",
]

IGNORE_VALUES = {
    "",
    "không",
    "không có",
    "không có thông tin",
    "không quy định",
    "chưa quy định",
    "không yêu cầu",
}

CONDITION_PATTERNS = [
    r"(?=\+\s*Trường hợp)",
    r"(?=Trường hợp)",
    r"(?=Đối với trường hợp)",
]


# =========================================================
# NORMALIZE / CLEAN
# =========================================================

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^\w\sàáạảãăắằẳẵặâấầẩẫậ"
        r"èéẹẻẽêếềểễệ"
        r"ìíịỉĩ"
        r"òóọỏõôốồổỗộơớờởỡợ"
        r"ùúụủũưứừửữự"
        r"ỳýỵỷỹđ]",
        "",
        text,
    )
    return text.strip()


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    for noise in NOISE_PHRASES:
        text = text.replace(noise, "")

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def should_ignore(value: str) -> bool:
    normalized = normalize_text(value)
    return normalized in IGNORE_VALUES


def slugify(text: str) -> str:
    return normalize_text(text).replace(" ", "_")


def text_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode()).hexdigest()[:12]


def dedupe_texts(items: List[str]) -> List[str]:
    result = []
    seen = set()
    for item in items:
        cleaned = clean_text(item)
        key = normalize_text(cleaned)
        if cleaned and key not in seen and not should_ignore(cleaned):
            result.append(cleaned)
            seen.add(key)
    return result


# =========================================================
# FORMAT VALUE
# =========================================================

def format_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return clean_text(value)

    if isinstance(value, list):
        lines = []
        for item in value:
            formatted = format_value(item)
            if formatted and not should_ignore(formatted):
                lines.append(f"- {formatted}")
        return "\n".join(lines)

    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            formatted = format_value(v)
            if formatted and not should_ignore(formatted):
                lines.append(f"{k}:\n{formatted}")
        return "\n".join(lines)

    return clean_text(str(value))


def split_conditions(text: str) -> List[str]:
    if not text:
        return []

    parts = re.split("|".join(CONDITION_PATTERNS), text)
    cleaned = []
    for part in parts:
        part = clean_text(part)
        if part:
            cleaned.append(part)
    return cleaned


# =========================================================
# METADATA / MARKDOWN
# =========================================================

def build_base_metadata(p_id: str, p_name: str, linh_vuc: str) -> Dict:
    return {
        "id": p_id,
        "procedure_id": p_id,
        "name": p_name,
        "procedure_name": p_name,
        "linh_vuc": linh_vuc,
        "lĩnh_vực": linh_vuc,
        "group_id": p_id,
    }


def build_chunk_text(
    p_name: str,
    linh_vuc: str,
    section: str,
    body: str,
    p_id: str = "",
) -> str:
    """
    Child chunk dạng Markdown có header rõ ràng.
    Header vẫn giữ các nhãn tiếng Việt để pipeline cũ có thể parse được.
    """
    header = [
        f"# Thủ tục: {p_name}",
    ]

    if p_id:
        header.append(f"- Mã thủ tục: {p_id}")

    header.extend([
        f"- Lĩnh vực: {linh_vuc}",
        f"- Mục thông tin: {section}",
        "",
        f"## {section}",
    ])

    return "\n".join(header).strip() + "\n" + clean_text(body)


def add_to_docs(chunk_text: str, metadata: Dict, docs: List[Document]):
    if should_ignore(chunk_text):
        return

    if len(chunk_text) <= MAX_CHARS:
        docs.append(Document(page_content=chunk_text, metadata=metadata))
        return

    splits = text_splitter.split_text(chunk_text)
    for idx, split in enumerate(splits, start=1):
        meta = copy.deepcopy(metadata)
        meta["chunk_part"] = idx
        meta["chunk_id"] = f'{meta["chunk_id"]}_part_{idx}'
        docs.append(Document(page_content=split, metadata=meta))


def build_procedure_markdown(item: Dict) -> str:
    """
    Parent context: toàn bộ thủ tục ở dạng Markdown.
    Dùng khi session đã chọn đúng thủ tục, giúp câu hỏi tiếp theo không cần dò toàn DB.
    """
    p_id = str(item.get("id", ""))
    p_name = item.get("name", "")
    content = item.get("content", {}) if isinstance(item.get("content", {}), dict) else {}
    linh_vuc = clean_text(content.get("Lĩnh vực", ""))

    lines = [
        f"# Thủ tục: {p_name}",
        f"- Mã thủ tục: {p_id}",
        f"- Lĩnh vực: {linh_vuc}",
    ]

    simple_fields = [
        "Tên thủ tục",
        "Đối tượng thực hiện",
        "Cơ quan thực hiện",
        "Cơ quan ban hành",
        "Cơ quan phối hợp",
        "Kết quả thực hiện",
        "Trình tự thực hiện",
        "Yêu cầu điều kiện",
        "Số bộ hồ sơ",
    ]

    for field in simple_fields:
        value = format_value(content.get(field, ""))
        if value and not should_ignore(value):
            lines.extend(["", f"## {field}", value])

    methods = content.get("Cách thức thực hiện", [])
    if isinstance(methods, list) and methods:
        methods = resolve_fee_links(methods, content.get("Phí", []), content.get("Lệ phí", []))
        method_lines = []
        seen = set()
        for idx, method in enumerate(methods, start=1):
            hinh_thuc = clean_text(method.get("Hình thức", ""))
            thoi_han = clean_text(method.get("Thời hạn", ""))
            mo_ta = clean_text(method.get("Mô tả", ""))
            fee = clean_text(method.get("Mức phí/Lệ phí", ""))

            key = normalize_text("|".join([hinh_thuc, thoi_han, mo_ta, fee]))
            if key in seen:
                continue
            seen.add(key)

            method_lines.append(f"### Cách thức {idx}: {hinh_thuc or 'Không ghi rõ'}")
            if thoi_han and not should_ignore(thoi_han):
                method_lines.append(f"- Thời hạn: {thoi_han}")
            if mo_ta and not should_ignore(mo_ta):
                method_lines.append(f"- Mô tả: {mo_ta}")
            if fee and not should_ignore(fee):
                method_lines.append(f"- Mức phí/Lệ phí: {fee}")
            method_lines.append("")

        if method_lines:
            lines.extend(["", "## Cách thức thực hiện", "\n".join(method_lines).strip()])

    documents = content.get("Thành phần hồ sơ", [])
    if isinstance(documents, list) and documents:
        doc_lines = []
        seen = set()
        for idx, doc_item in enumerate(documents, start=1):
            ten_giay_to = clean_text(doc_item.get("Tên giấy tờ", ""))
            bieu_mau = clean_text(doc_item.get("Biểu mẫu", ""))
            so_luong = clean_text(doc_item.get("Số lượng", ""))

            key = normalize_text("|".join([ten_giay_to, bieu_mau, so_luong]))
            if should_ignore(ten_giay_to) or key in seen:
                continue
            seen.add(key)

            doc_lines.append(f"{len(seen)}. {ten_giay_to}")
            if bieu_mau and not should_ignore(bieu_mau):
                doc_lines.append(f"   - Biểu mẫu: {bieu_mau}")
            if so_luong and not should_ignore(so_luong):
                doc_lines.append(f"   - Số lượng: {so_luong}")

        if doc_lines:
            lines.extend(["", "## Thành phần hồ sơ", "\n".join(doc_lines)])

    fee_texts = []
    for field in ["Phí", "Lệ phí"]:
        raw_list = content.get(field, [])
        if isinstance(raw_list, list):
            for item_fee in raw_list:
                if isinstance(item_fee, dict):
                    fee_texts.append(clean_text(item_fee.get("text", "")))
                else:
                    fee_texts.append(clean_text(str(item_fee)))
        else:
            fee_texts.append(format_value(raw_list))
    fee_texts = dedupe_texts(fee_texts)
    if fee_texts:
        lines.extend(["", "## Phí/Lệ phí", "\n".join(f"- {x}" for x in fee_texts)])

    legal_docs = content.get("Căn cứ pháp lý", [])
    if isinstance(legal_docs, list) and legal_docs:
        legal_lines = []
        seen = set()
        for law in legal_docs:
            so_hieu = clean_text(law.get("Số hiệu", ""))
            ten_vb = clean_text(law.get("Tên văn bản", ""))
            ngay_bh = clean_text(law.get("Ngày ban hành", ""))
            co_quan = clean_text(law.get("Cơ quan ban hành", ""))
            key = normalize_text("|".join([so_hieu, ten_vb, ngay_bh, co_quan]))
            if key in seen:
                continue
            seen.add(key)
            legal_lines.append(f"- Số hiệu: {so_hieu}")
            if ten_vb:
                legal_lines.append(f"  Tên văn bản: {ten_vb}")
            if ngay_bh:
                legal_lines.append(f"  Ngày ban hành: {ngay_bh}")
            if co_quan and not should_ignore(co_quan):
                legal_lines.append(f"  Cơ quan ban hành: {co_quan}")

        if legal_lines:
            lines.extend(["", "## Căn cứ pháp lý", "\n".join(legal_lines)])

    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# =========================================================
# RESOLVE FEES
# =========================================================

def resolve_fee_links(methods: List[Dict], fee_list: List[Dict], lephi_list: List[Dict]) -> List[Dict]:
    methods_copy = copy.deepcopy(methods)

    fee_dict = {
        f["id"]: clean_text(f.get("text", ""))
        for f in fee_list
        if isinstance(f, dict) and "id" in f
    }
    lephi_dict = {
        f["id"]: clean_text(f.get("text", ""))
        for f in lephi_list
        if isinstance(f, dict) and "id" in f
    }
    all_fees = {**fee_dict, **lephi_dict}

    fee_cache = {}
    for method in methods_copy:
        links = method.get("Liên kết phí", [])
        fee_texts = []
        for link in links:
            fee_text = all_fees.get(link)
            if fee_text:
                fee_hash = text_hash(fee_text)
                if fee_hash not in fee_cache:
                    fee_cache[fee_hash] = fee_text
                fee_texts.append(fee_cache[fee_hash])

        fee_texts = list(dict.fromkeys(fee_texts))
        if fee_texts:
            method["Mức phí/Lệ phí"] = "\n".join(fee_texts)
        method.pop("Liên kết phí", None)

    return methods_copy


# =========================================================
# CHUNK CREATORS
# =========================================================

def create_full_procedure_chunk(item: Dict, docs: List[Document]):
    p_id = str(item.get("id", ""))
    p_name = item.get("name", "")
    content = item.get("content", {}) if isinstance(item.get("content", {}), dict) else {}
    linh_vuc = clean_text(content.get("Lĩnh vực", ""))

    chunk_text = build_procedure_markdown(item)
    metadata = build_base_metadata(p_id, p_name, linh_vuc)
    metadata.update({
        "field": "Toàn bộ thủ tục",
        "section_type": "procedure_full",
        "priority": 20,
        "chunk_id": f"{p_id}_procedure_full",
        "is_parent_context": True,
    })
    add_to_docs(chunk_text, metadata, docs)


def create_summary_chunk(item: Dict, docs: List[Document]):
    p_id = str(item.get("id", ""))
    p_name = item.get("name", "")
    content = item.get("content", {})
    linh_vuc = clean_text(content.get("Lĩnh vực", ""))

    summary_lines = []
    fields = ["Đối tượng thực hiện", "Cơ quan thực hiện", "Kết quả thực hiện"]
    for field in fields:
        value = format_value(content.get(field, ""))
        if value and not should_ignore(value):
            summary_lines.append(f"{field}: {value}")

    methods = content.get("Cách thức thực hiện", [])
    if methods:
        method_names = []
        for method in methods:
            hinh_thuc = clean_text(method.get("Hình thức", ""))
            if hinh_thuc:
                method_names.append(hinh_thuc)
        method_names = dedupe_texts(method_names)
        if method_names:
            summary_lines.append(f"Hỗ trợ: {', '.join(method_names)}")

    if not summary_lines:
        return

    chunk_text = build_chunk_text(p_name, linh_vuc, "Tổng quan", "\n".join(summary_lines), p_id=p_id)
    metadata = build_base_metadata(p_id, p_name, linh_vuc)
    metadata.update({
        "field": "summary",
        "section_type": "summary",
        "priority": 10,
        "chunk_id": f"{p_id}_summary",
    })
    add_to_docs(chunk_text, metadata, docs)


def create_method_chunks(item: Dict, docs: List[Document]):
    p_id = str(item.get("id", ""))
    p_name = item.get("name", "")
    content = item.get("content", {})
    linh_vuc = clean_text(content.get("Lĩnh vực", ""))
    methods = content.get("Cách thức thực hiện", [])

    if not isinstance(methods, list):
        return

    methods = resolve_fee_links(methods, content.get("Phí", []), content.get("Lệ phí", []))
    seen_methods = set()

    for idx, method in enumerate(methods, start=1):
        hinh_thuc = clean_text(method.get("Hình thức", ""))
        thoi_han = clean_text(method.get("Thời hạn", ""))
        mo_ta = clean_text(method.get("Mô tả", ""))
        fee = clean_text(method.get("Mức phí/Lệ phí", ""))

        condition_parts = split_conditions(thoi_han) or [thoi_han]
        for c_idx, condition in enumerate(condition_parts, start=1):
            key = normalize_text("|".join([hinh_thuc, condition, mo_ta, fee]))
            if key in seen_methods:
                continue
            seen_methods.add(key)

            body = (
                f"Hình thức: {hinh_thuc}\n\n"
                f"Thời hạn:\n{condition}\n\n"
                f"Mô tả:\n{mo_ta}"
            )
            if fee and not should_ignore(fee):
                body += f"\n\nMức phí/Lệ phí:\n{fee}"

            chunk_text = build_chunk_text(p_name, linh_vuc, "Cách thức thực hiện", body, p_id=p_id)
            metadata = build_base_metadata(p_id, p_name, linh_vuc)
            metadata.update({
                "field": "Cách thức thực hiện",
                "section_type": "method",
                "priority": 9,
                "method_type": hinh_thuc,
                "parent_chunk_id": f"{p_id}_summary",
                "chunk_id": f"{p_id}_method_{idx}_condition_{c_idx}",
                "semantic_tags": ["cách thức", hinh_thuc.lower()],
            })
            add_to_docs(chunk_text, metadata, docs)


def create_document_chunks(item: Dict, docs: List[Document]):
    p_id = str(item.get("id", ""))
    p_name = item.get("name", "")
    content = item.get("content", {})
    linh_vuc = clean_text(content.get("Lĩnh vực", ""))
    documents = content.get("Thành phần hồ sơ", [])

    if not isinstance(documents, list):
        return

    total = len(documents)
    seen_docs = set()
    logical_idx = 0

    for idx, doc_item in enumerate(documents, start=1):
        ten_giay_to = clean_text(doc_item.get("Tên giấy tờ", ""))
        if should_ignore(ten_giay_to):
            continue

        bieu_mau = clean_text(doc_item.get("Biểu mẫu", ""))
        so_luong = clean_text(doc_item.get("Số lượng", ""))
        key = normalize_text("|".join([ten_giay_to, bieu_mau, so_luong]))
        if key in seen_docs:
            continue
        seen_docs.add(key)
        logical_idx += 1

        body = (
            f"Hồ sơ số: {logical_idx}/{total}\n\n"
            f"Tên giấy tờ:\n{ten_giay_to}"
        )
        if bieu_mau and not should_ignore(bieu_mau):
            body += f"\n\nBiểu mẫu:\n{bieu_mau}"
        if so_luong and not should_ignore(so_luong):
            body += f"\n\nSố lượng:\n{so_luong}"

        chunk_text = build_chunk_text(p_name, linh_vuc, "Thành phần hồ sơ", body, p_id=p_id)
        metadata = build_base_metadata(p_id, p_name, linh_vuc)
        metadata.update({
            "field": "Thành phần hồ sơ",
            "section_type": "document",
            "priority": 10,
            "parent_chunk_id": f"{p_id}_summary",
            "chunk_id": f"{p_id}_document_{idx}",
            "semantic_tags": ["hồ sơ", "giấy tờ"],
        })
        add_to_docs(chunk_text, metadata, docs)


def create_legal_chunks(item: Dict, docs: List[Document]):
    p_id = str(item.get("id", ""))
    p_name = item.get("name", "")
    content = item.get("content", {})
    linh_vuc = clean_text(content.get("Lĩnh vực", ""))
    legal_docs = content.get("Căn cứ pháp lý", [])

    if not isinstance(legal_docs, list):
        return

    seen_laws = set()
    logical_idx = 0

    for idx, law in enumerate(legal_docs, start=1):
        so_hieu = clean_text(law.get("Số hiệu", ""))
        ten_vb = clean_text(law.get("Tên văn bản", ""))
        ngay_bh = clean_text(law.get("Ngày ban hành", ""))
        co_quan = clean_text(law.get("Cơ quan ban hành", ""))

        key = normalize_text("|".join([so_hieu, ten_vb, ngay_bh, co_quan]))
        if key in seen_laws:
            continue
        seen_laws.add(key)
        logical_idx += 1

        body = (
            f"Văn bản pháp lý số: {logical_idx}\n"
            f"Số hiệu: {so_hieu}\n"
            f"Tên văn bản: {ten_vb}\n"
            f"Ngày ban hành: {ngay_bh}\n"
            f"Cơ quan ban hành: {co_quan}"
        )

        chunk_text = build_chunk_text(p_name, linh_vuc, "Căn cứ pháp lý", body, p_id=p_id)
        metadata = build_base_metadata(p_id, p_name, linh_vuc)
        metadata.update({
            "field": "Căn cứ pháp lý",
            "section_type": "legal",
            "priority": 5,
            "law_id": so_hieu,
            "parent_chunk_id": f"{p_id}_summary",
            "chunk_id": f"{p_id}_legal_{idx}",
            "semantic_tags": ["pháp lý", so_hieu.lower()],
        })
        add_to_docs(chunk_text, metadata, docs)


def create_simple_field_chunks(item: Dict, docs: List[Document]):
    simple_fields = [
        "Trình tự thực hiện",
        "Đối tượng thực hiện",
        "Cơ quan thực hiện",
        "Kết quả thực hiện",
        "Yêu cầu điều kiện",
        "Thời hạn giải quyết",
        "Phí",
        "Lệ phí",
        "Cơ quan ban hành",
        "Cơ quan phối hợp",
        "Số bộ hồ sơ",
    ]

    p_id = str(item.get("id", ""))
    p_name = item.get("name", "")
    content = item.get("content", {})
    linh_vuc = clean_text(content.get("Lĩnh vực", ""))

    for field in simple_fields:
        raw_value = content.get(field, "")

        if field in ["Phí", "Lệ phí"] and isinstance(raw_value, list):
            texts = []
            for f in raw_value:
                if isinstance(f, dict) and "text" in f:
                    texts.append(clean_text(f["text"]))
                elif isinstance(f, str):
                    texts.append(clean_text(f))
            value = "\n".join(dedupe_texts(texts))
        else:
            value = format_value(raw_value)

        if should_ignore(value):
            continue

        chunk_text = build_chunk_text(p_name, linh_vuc, field, value, p_id=p_id)
        metadata = build_base_metadata(p_id, p_name, linh_vuc)
        metadata.update({
            "field": field,
            "section_type": "simple",
            "priority": 7,
            "parent_chunk_id": f"{p_id}_summary",
            "chunk_id": f"{p_id}_{slugify(field)}",
            "semantic_tags": [slugify(field)],
        })
        add_to_docs(chunk_text, metadata, docs)


# =========================================================
# MAIN
# =========================================================

def create_chunks(data: List[Dict]) -> List[Document]:
    docs = []

    for item in data:
        try:
            create_full_procedure_chunk(item, docs)
            create_summary_chunk(item, docs)
            create_method_chunks(item, docs)
            create_document_chunks(item, docs)
            create_legal_chunks(item, docs)
            create_simple_field_chunks(item, docs)
        except Exception as e:
            print(f"[CHUNK ERROR] {item.get('id')} -> {e}")

    print(f"[CHUNKER] Created {len(docs)} semantic/contextual chunks.")
    return docs
