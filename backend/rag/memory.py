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
    Chuyển lịch sử thành chuỗi văn bản để nạp vào Prompt của LLM.
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


def get_selected_procedure(session_id: str):
    """
    Lấy thủ tục đang được chọn trong phiên chat.
    Trả về None nếu session chưa có thủ tục chính hoặc Firestore chưa sẵn sàng.
    """
    try:
        db = get_db()
        doc = db.collection('sessions').document(session_id).get()
        if not doc.exists:
            return None

        data = doc.to_dict() or {}
        procedure_id = data.get("selected_procedure_id")
        procedure_name = data.get("selected_procedure_name")

        if not procedure_id and not procedure_name:
            return None

        return {
            "id": procedure_id,
            "name": procedure_name,
            "score": data.get("selected_procedure_score"),
            "last_field": data.get("selected_last_field"),
            "updated_at": data.get("selected_updated_at"),
        }
    except Exception as e:
        print(f"[FIRESTORE WARNING] Không lấy được selected procedure: {e}")
        return None


def save_selected_procedure(
    session_id: str,
    procedure_id: str = "",
    procedure_name: str = "",
    score=None,
    field=None,
):
    """
    Lưu thủ tục chính của phiên chat để các câu hỏi tiếp theo dùng lại context này.
    Không lưu full context vào Firestore để tránh phình dữ liệu.
    """
    if not session_id or not procedure_name:
        return

    try:
        db = get_db()
        payload = {
            "selected_procedure_id": procedure_id or "",
            "selected_procedure_name": procedure_name,
            "selected_updated_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }

        if score is not None:
            try:
                payload["selected_procedure_score"] = float(score)
            except Exception:
                payload["selected_procedure_score"] = score

        if field:
            if isinstance(field, list):
                payload["selected_last_field"] = field[0] if field else ""
            else:
                payload["selected_last_field"] = field

        db.collection('sessions').document(session_id).set(payload, merge=True)
    except Exception as e:
        print(f"[FIRESTORE WARNING] Không lưu được selected procedure: {e}")


def clear_selected_procedure(session_id: str):
    """Xóa thủ tục đang chọn nhưng giữ lại phiên chat."""
    try:
        db = get_db()
        db.collection('sessions').document(session_id).update({
            "selected_procedure_id": firestore.DELETE_FIELD,
            "selected_procedure_name": firestore.DELETE_FIELD,
            "selected_procedure_score": firestore.DELETE_FIELD,
            "selected_last_field": firestore.DELETE_FIELD,
            "selected_updated_at": firestore.DELETE_FIELD,
        })
    except Exception as e:
        print(f"[FIRESTORE WARNING] Không xóa được selected procedure: {e}")


def save_message(uid: str, session_id: str, role: str, content: str):
    """Lưu tin nhắn vào Firestore"""
    try:
        db = get_db()
        session_ref = db.collection('sessions').document(session_id)

        # 1. Cập nhật thông tin của Session (Phiên chat)
        session_doc = session_ref.get()

        session_data = {
            "uid": uid,
            "updated_at": firestore.SERVER_TIMESTAMP,
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
            "timestamp": firestore.SERVER_TIMESTAMP,
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
