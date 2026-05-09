from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def format_field_value(val):
    """Xử lý dữ liệu không đồng nhất: lúc thì string, lúc thì list dict"""
    if not val or val == "Không có thông tin":
        return None
        
    if isinstance(val, str):
        return val.strip()
        
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, dict):
                pairs = []
                if "Tên giấy tờ" in item: pairs.append(str(item["Tên giấy tờ"]))
                if "Số lượng" in item: pairs.append(f"(Số lượng: {item['Số lượng']})")
                if "Tên văn bản" in item: pairs.append(str(item["Tên văn bản"]))
                if "Số hiệu" in item: pairs.append(f"(Số hiệu: {item['Số hiệu']})")
                
                if not pairs:
                    pairs = [f"{k}: {v}" for k, v in item.items() if v]
                
                parts.append("- " + " ".join(pairs))
            elif isinstance(item, str):
                parts.append("- " + item)
        return "\n".join(parts)
        
    return str(val).strip()

def create_enriched_documents(base_meta, field_name, content_body):
    """
    Hàm này ép tên thủ tục vào nội dung và chia nhỏ văn bản (Chunking).
    """
    if not content_body:
        return []
        
    prefix = f"THỦ TỤC: {base_meta['ten_thu_tuc']}\nMỤC: {field_name}\nNỘI DUNG: "
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    splits = text_splitter.split_text(content_body)
    docs = []
    
    for split in splits:
        docs.append(Document(
            page_content=prefix + split,
            metadata={
                "name": base_meta["ten_thu_tuc"],
                "field": field_name,
                "id": base_meta["id"]
            }
        ))

    return docs

def create_chunks(data_list):
    documents = []
    
    for item in data_list:
        content = item.get("content", {})
        base_meta = {
            "id": item.get("id"),
            "ten_thu_tuc": item.get("name")
        }

        fields = [
            "Đối tượng thực hiện", "Cơ quan thực hiện", 
            "Kết quả thực hiện", "Trình tự thực hiện", "Cách thức thực hiện",
            "Thành phần hồ sơ", "Lệ phí", "Phí", "Căn cứ pháp lý",
            "Yêu cầu điều kiện", "Thời hạn giải quyết"
        ]
        
        for f in fields:
            val = format_field_value(content.get(f))
            if val:
                docs = create_enriched_documents(base_meta, f, val)
                documents.extend(docs)

    return documents