from firebase_admin import firestore

MAX_TURNS = 10  # Lưu 10 tin nhắn gần nhất (tương đương 5 lượt hỏi-đáp)

def get_db():
    return firestore.client()

def get_history(session_id: str):
    """Lấy lịch sử dạng danh sách object cho API từ Firestore"""
    db = get_db()
    docs = db.collection('sessions').document(session_id).collection('messages')\
             .order_by('timestamp', direction=firestore.Query.DESCENDING)\
             .limit(MAX_TURNS).stream()
    
    history = []
    for doc in docs:
        data = doc.to_dict()
        history.append({"role": data.get("role"), "content": data.get("content")})
    
    # Đảo ngược lại để đúng thứ tự thời gian (cũ -> mới) cho LLM đọc
    history.reverse()
    return history

def get_history_as_string(session_id: str):
    """
    Chuyển lịch sử thành chuỗi văn bản để nạp vào Prompt của LLM
    Giúp LLM hiểu ngữ cảnh câu hỏi trước đó.
    """
    history = get_history(session_id)
    if not history:
        return ""
    
    formatted_history = []
    for msg in history:
        role = "Người dùng" if msg["role"] == "user" else "Bot"
        formatted_history.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted_history)

def save_message(uid: str, session_id: str, role: str, content: str):
    """Lưu tin nhắn vào Firestore"""
    try:
        db = get_db()
        session_ref = db.collection('sessions').document(session_id)
        
        # 1. Cập nhật thông tin của Session (Phiên chat)
        session_doc = session_ref.get()
        
        session_data = {
            "uid": uid,
            "updated_at": firestore.SERVER_TIMESTAMP
        }
        
        if not session_doc.exists:
            # Nếu là tin nhắn đầu tiên, tạo title từ nội dung câu hỏi
            session_data["created_at"] = firestore.SERVER_TIMESTAMP
            if role == "user":
                # Lấy 30 ký tự đầu làm tiêu đề
                session_data["title"] = content[:30] + "..." if len(content) > 30 else content
            else:
                session_data["title"] = "Trò chuyện mới"
            session_ref.set(session_data)
        else:
            # Nếu session đã tồn tại, chỉ cập nhật thời gian
            session_ref.update(session_data)

        # 2. Thêm tin nhắn vào Sub-collection
        session_ref.collection('messages').add({
            "role": role,
            "content": content,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"[FIRESTORE ERROR] Lỗi lưu tin nhắn vào DB: {e}")

def clear_history(session_id: str):
    """Xóa lịch sử trong Firestore"""
    db = get_db()
    # Xóa các tin nhắn bên trong
    docs = db.collection('sessions').document(session_id).collection('messages').stream()
    for doc in docs:
        doc.reference.delete()
        
    # Xóa luôn document của session đó
    db.collection('sessions').document(session_id).delete()