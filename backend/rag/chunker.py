from langchain_core.documents import Document


def safe_get(item, key):
    value = item.get(key)
    if not value or str(value).strip() == "":
        return "Không có thông tin"
    return str(value)


FIELDS = [
    "Tên thủ tục",
    "Đối tượng thực hiện",
    "Yêu cầu điều kiện",
    "Cách thức thực hiện",
    "Trình tự thực hiện",
    "Thành phần hồ sơ",
    "Số bộ hồ sơ",
    "Thời hạn giải quyết",
    "Kết quả thực hiện",
    "Phí",
    "Lệ phí",
    "Căn cứ pháp lý",
    "Cơ quan thực hiện",
    "Cơ quan ban hành",
    "Cơ quan phối hợp",
    "Đầu mối hỗ trợ"
]


def create_chunks(data):
    documents = []

    for item in data:
        content_data = item.get("content", {})

        if not content_data:
            continue

        ten_thu_tuc = safe_get(content_data, "Tên thủ tục")

        for field in FIELDS:
            value = safe_get(content_data, field)

            if value == "Không có thông tin":
                continue

            content = f"""
TÊN THỦ TỤC: {ten_thu_tuc}
LOẠI THÔNG TIN: {field}

NỘI DUNG:
{value}
"""

            doc = Document(
                page_content=content,
                metadata={
                    "ten_thu_tuc": ten_thu_tuc,
                    "field": field
                }
            )

            documents.append(doc)

    print(f"Created {len(documents)} chunks")

    if documents:
        print("=== SAMPLE CHUNK ===")
        print(documents[0].page_content[:500])
    else:
        print("❌ No documents created - check data format!")

    return documents
