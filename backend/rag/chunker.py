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

# Giữ chunk nhỏ theo khuyến nghị RAG: khoảng 300-800 tokens/chunk.
# Ước lượng tiếng Việt đơn giản: 1 token ≈ 4 ký tự.
MAX_CHARS = 4500

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=400,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
    ]
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
# NORMALIZE
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


# =========================================================
# CLEAN
# =========================================================

def clean_text(text: Any) -> str:

    if not isinstance(text, str):
        return ""

    for noise in NOISE_PHRASES:
        text = text.replace(noise, "")

    text = re.sub(r"\n\s*\n", "\n", text)

    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# =========================================================
# IGNORE CHECK
# =========================================================

def should_ignore(value: str) -> bool:

    normalized = normalize_text(value)

    return normalized in IGNORE_VALUES


# =========================================================
# SLUGIFY
# =========================================================

def slugify(text: str) -> str:

    text = normalize_text(text)

    text = text.replace(" ", "_")

    return text


# =========================================================
# HASH TEXT
# =========================================================

def text_hash(text: str) -> str:

    return hashlib.md5(
        normalize_text(text).encode()
    ).hexdigest()[:12]


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

                lines.append(
                    f"{k}:\n{formatted}"
                )

        return "\n".join(lines)

    return clean_text(str(value))


# =========================================================
# SPLIT CONDITIONS
# =========================================================

def split_conditions(text: str) -> List[str]:

    if not text:
        return []

    parts = re.split(
        "|".join(CONDITION_PATTERNS),
        text
    )

    cleaned = []

    for part in parts:

        part = clean_text(part)

        if part:
            cleaned.append(part)

    return cleaned


# =========================================================
# BUILD METADATA
# =========================================================

def build_base_metadata(
    p_id: str,
    p_name: str,
    linh_vuc: str,
) -> Dict:

    return {
        "id": p_id,
        "procedure_id": p_id,

        "name": p_name,
        "procedure_name": p_name,

        "linh_vuc": linh_vuc,
        "lĩnh_vực": linh_vuc,

        "group_id": p_id,
    }


# =========================================================
# BUILD CHUNK TEXT
# =========================================================

def build_chunk_text(
    p_name: str,
    linh_vuc: str,
    section: str,
    body: str,
) -> str:

    return (
        f"Thủ tục: {p_name}\n"
        f"Lĩnh vực: {linh_vuc}\n"
        f"Loại thông tin: {section}\n\n"
        f"{body}"
    )


# =========================================================
# ADD DOCS
# =========================================================

def add_to_docs(
    chunk_text: str,
    metadata: Dict,
    docs: List[Document]
):

    if should_ignore(chunk_text):
        return

    # Không split semantic chunk nhỏ
    if len(chunk_text) <= MAX_CHARS:

        docs.append(
            Document(
                page_content=chunk_text,
                metadata=metadata
            )
        )

        return

    splits = text_splitter.split_text(chunk_text)

    for idx, split in enumerate(splits, start=1):

        meta = copy.deepcopy(metadata)

        meta["chunk_part"] = idx

        meta["chunk_id"] = (
            f'{meta["chunk_id"]}_part_{idx}'
        )

        docs.append(
            Document(
                page_content=split,
                metadata=meta
            )
        )


# =========================================================
# RESOLVE FEES
# =========================================================

def resolve_fee_links(
    methods: List[Dict],
    fee_list: List[Dict],
    lephi_list: List[Dict],
) -> List[Dict]:

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

    all_fees = {
        **fee_dict,
        **lephi_dict,
    }

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

                fee_texts.append(
                    fee_cache[fee_hash]
                )

        fee_texts = list(dict.fromkeys(fee_texts))

        if fee_texts:

            method["Mức phí/Lệ phí"] = (
                "\n".join(fee_texts)
            )

        method.pop("Liên kết phí", None)

    return methods_copy


# =========================================================
# SUMMARY CHUNK
# =========================================================

def create_summary_chunk(
    item: Dict,
    docs: List[Document]
):

    p_id = item.get("id", "")

    p_name = item.get("name", "")

    content = item.get("content", {})

    linh_vuc = content.get(
        "Lĩnh vực",
        ""
    )

    summary_lines = []

    fields = [
        "Đối tượng thực hiện",
        "Cơ quan thực hiện",
        "Kết quả thực hiện",
    ]

    for field in fields:

        value = format_value(
            content.get(field, "")
        )

        if value and not should_ignore(value):

            summary_lines.append(
                f"{field}: {value}"
            )

    methods = content.get(
        "Cách thức thực hiện",
        []
    )

    if methods:

        method_names = []

        for method in methods:

            hinh_thuc = clean_text(
                method.get("Hình thức", "")
            )

            if hinh_thuc:
                method_names.append(hinh_thuc)

        if method_names:

            summary_lines.append(
                f"Hỗ trợ: {', '.join(method_names)}"
            )

    chunk_text = build_chunk_text(
        p_name,
        linh_vuc,
        "Tổng quan",
        "\n".join(summary_lines)
    )

    metadata = build_base_metadata(
        p_id,
        p_name,
        linh_vuc
    )

    metadata.update({
        "field": "summary",
        "section_type": "summary",
        "priority": 10,
        "chunk_id": f"{p_id}_summary",
    })

    add_to_docs(
        chunk_text,
        metadata,
        docs
    )


# =========================================================
# METHOD CHUNKS
# =========================================================

def create_method_chunks(
    item: Dict,
    docs: List[Document]
):

    p_id = item.get("id", "")

    p_name = item.get("name", "")

    content = item.get("content", {})

    linh_vuc = content.get(
        "Lĩnh vực",
        ""
    )

    methods = content.get(
        "Cách thức thực hiện",
        []
    )

    if not isinstance(methods, list):
        return

    methods = resolve_fee_links(
        methods,
        content.get("Phí", []),
        content.get("Lệ phí", [])
    )

    for idx, method in enumerate(methods, start=1):

        hinh_thuc = clean_text(
            method.get("Hình thức", "")
        )

        thoi_han = clean_text(
            method.get("Thời hạn", "")
        )

        mo_ta = clean_text(
            method.get("Mô tả", "")
        )

        fee = clean_text(
            method.get("Mức phí/Lệ phí", "")
        )

        # Split condition
        condition_parts = split_conditions(
            thoi_han
        )

        if not condition_parts:
            condition_parts = [thoi_han]

        for c_idx, condition in enumerate(
            condition_parts,
            start=1
        ):

            body = (
                f"Hình thức: {hinh_thuc}\n\n"
                f"Thời hạn:\n{condition}\n\n"
                f"Mô tả:\n{mo_ta}"
            )

            if fee and not should_ignore(fee):

                body += (
                    f"\n\nMức phí/Lệ phí:\n"
                    f"{fee}"
                )

            chunk_text = build_chunk_text(
                p_name,
                linh_vuc,
                "Cách thức thực hiện",
                body
            )

            metadata = build_base_metadata(
                p_id,
                p_name,
                linh_vuc
            )

            metadata.update({

                "field": "Cách thức thực hiện",

                "section_type": "method",

                "priority": 9,

                "method_type": hinh_thuc,

                "parent_chunk_id":
                    f"{p_id}_summary",

                "chunk_id":
                    f"{p_id}_method_{idx}_condition_{c_idx}",

                "semantic_tags": [
                    "cách thức",
                    hinh_thuc.lower(),
                ]
            })

            add_to_docs(
                chunk_text,
                metadata,
                docs
            )


# =========================================================
# DOCUMENT CHUNKS
# =========================================================

def create_document_chunks(
    item: Dict,
    docs: List[Document]
):

    p_id = item.get("id", "")

    p_name = item.get("name", "")

    content = item.get("content", {})

    linh_vuc = content.get(
        "Lĩnh vực",
        ""
    )

    documents = content.get(
        "Thành phần hồ sơ",
        []
    )

    if not isinstance(documents, list):
        return

    for idx, doc_item in enumerate(
        documents,
        start=1
    ):

        ten_giay_to = clean_text(
            doc_item.get(
                "Tên giấy tờ",
                ""
            )
        )

        if should_ignore(ten_giay_to):
            continue

        bieu_mau = clean_text(
            doc_item.get(
                "Biểu mẫu",
                ""
            )
        )

        so_luong = clean_text(
            doc_item.get(
                "Số lượng",
                ""
            )
        )

        body = (
            f"Tên giấy tờ:\n"
            f"{ten_giay_to}"
        )

        if bieu_mau:

            body += (
                f"\n\nBiểu mẫu:\n"
                f"{bieu_mau}"
            )

        if so_luong:

            body += (
                f"\n\nSố lượng:\n"
                f"{so_luong}"
            )

        chunk_text = build_chunk_text(
            p_name,
            linh_vuc,
            "Thành phần hồ sơ",
            body
        )

        metadata = build_base_metadata(
            p_id,
            p_name,
            linh_vuc
        )

        metadata.update({

            "field": "Thành phần hồ sơ",

            "section_type": "document",

            "priority": 10,

            "parent_chunk_id":
                f"{p_id}_summary",

            "chunk_id":
                f"{p_id}_document_{idx}",

            "semantic_tags": [
                "hồ sơ",
                "giấy tờ",
            ]
        })

        add_to_docs(
            chunk_text,
            metadata,
            docs
        )


# =========================================================
# LEGAL CHUNKS
# =========================================================

def create_legal_chunks(
    item: Dict,
    docs: List[Document]
):

    p_id = item.get("id", "")

    p_name = item.get("name", "")

    content = item.get("content", {})

    linh_vuc = content.get(
        "Lĩnh vực",
        ""
    )

    legal_docs = content.get(
        "Căn cứ pháp lý",
        []
    )

    if not isinstance(legal_docs, list):
        return

    for idx, law in enumerate(
        legal_docs,
        start=1
    ):

        so_hieu = clean_text(
            law.get("Số hiệu", "")
        )

        ten_vb = clean_text(
            law.get("Tên văn bản", "")
        )

        ngay_bh = clean_text(
            law.get("Ngày ban hành", "")
        )

        co_quan = clean_text(
            law.get(
                "Cơ quan ban hành",
                ""
            )
        )

        body = (
            f"Số hiệu: {so_hieu}\n"
            f"Tên văn bản: {ten_vb}\n"
            f"Ngày ban hành: {ngay_bh}\n"
            f"Cơ quan ban hành: {co_quan}"
        )

        chunk_text = build_chunk_text(
            p_name,
            linh_vuc,
            "Căn cứ pháp lý",
            body
        )

        metadata = build_base_metadata(
            p_id,
            p_name,
            linh_vuc
        )

        metadata.update({

            "field": "Căn cứ pháp lý",

            "section_type": "legal",

            "priority": 5,

            "law_id": so_hieu,

            "parent_chunk_id":
                f"{p_id}_summary",

            "chunk_id":
                f"{p_id}_legal_{idx}",

            "semantic_tags": [
                "pháp lý",
                so_hieu.lower(),
            ]
        })

        add_to_docs(
            chunk_text,
            metadata,
            docs
        )


# =========================================================
# SIMPLE FIELD CHUNKS
# =========================================================

def create_simple_field_chunks(
    item: Dict,
    docs: List[Document]
):

    SIMPLE_FIELDS = [

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

    p_id = item.get("id", "")

    p_name = item.get("name", "")

    content = item.get("content", {})

    linh_vuc = content.get(
        "Lĩnh vực",
        ""
    )

    for field in SIMPLE_FIELDS:

        raw_value = content.get(field, "")

        if field in ["Phí", "Lệ phí"] and isinstance(raw_value, list):
            texts = []
            for f in raw_value:
                if isinstance(f, dict) and "text" in f:
                    texts.append(clean_text(f["text"]))
                elif isinstance(f, str):
                    texts.append(clean_text(f))
            texts = list(dict.fromkeys(texts))
            value = "\n".join(texts)
        else:
            value = format_value(raw_value)

        if should_ignore(value):
            continue

        chunk_text = build_chunk_text(
            p_name,
            linh_vuc,
            field,
            value
        )

        metadata = build_base_metadata(
            p_id,
            p_name,
            linh_vuc
        )

        metadata.update({

            "field": field,

            "section_type": "simple",

            "priority": 7,

            "parent_chunk_id":
                f"{p_id}_summary",

            "chunk_id":
                f"{p_id}_{slugify(field)}",

            "semantic_tags": [
                slugify(field)
            ]
        })

        add_to_docs(
            chunk_text,
            metadata,
            docs
        )


# =========================================================
# MAIN
# =========================================================

def create_chunks(
    data: List[Dict]
) -> List[Document]:

    docs = []

    for item in data:

        try:

            create_summary_chunk(
                item,
                docs
            )

            create_method_chunks(
                item,
                docs
            )

            create_document_chunks(
                item,
                docs
            )

            create_legal_chunks(
                item,
                docs
            )

            create_simple_field_chunks(
                item,
                docs
            )

        except Exception as e:

            print(
                f"[CHUNK ERROR] "
                f'{item.get("id")} -> {e}'
            )

    print(
        f"[CHUNKER] "
        f"Created {len(docs)} semantic chunks."
    )

    return docs