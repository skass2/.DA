from fastapi import APIRouter, Request, Depends
from rag.pipeline import ask_rag
from rag.memory import get_history, save_message, clear_history, get_db
from routers.auth import get_current_user

router = APIRouter(prefix="/user", tags=["User"])

# API Chat chuyển từ main.py sang, có thêm dependency get_current_user
@router.get("/chat")
def chat(request: Request, q: str, session_id: str = "default", current_user: dict = Depends(get_current_user)):
    
    if not q or q.strip() == "":
        return {"answer": "Vui lòng nhập câu hỏi."}

    uid = current_user.get('uid')

    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"answer": "Hệ thống đang khởi tạo dữ liệu, vui lòng đợi trong giây lát."}

    history = get_history(session_id)
    answer = ask_rag(db=db, query=q, session_id=session_id, history=history)

    # Đẩy UID và Session ID vào DB
    save_message(uid, session_id, "user", q)
    save_message(uid, session_id, "bot", answer)

    return {"answer": answer}

# ===== API LẤY DANH SÁCH LỊCH SỬ CHAT CHO SIDEBAR =====
@router.get("/sessions")
def get_user_sessions(current_user: dict = Depends(get_current_user)):
    uid = current_user.get('uid')
    db = get_db()
    
    # Truy vấn tất cả session của User này, xếp cái nào mới chat lên đầu
    docs = db.collection('sessions').where('uid', '==', uid)\
             .order_by('updated_at', direction='DESCENDING').stream()
             
    sessions = []
    for doc in docs:
        data = doc.to_dict()
        sessions.append({
            "session_id": doc.id,
            "title": data.get("title", "Trò chuyện mới"),
            # Format lại thời gian nếu cần thiết
            "updated_at": data.get("updated_at")
        })
        
    return {"sessions": sessions}
# ===== API LẤY LỊCH SỬ TIN NHẮN CỦA 1 SESSION ĐỂ HIỂN THỊ KHI CLICK =====
@router.get("/chat/history")
def get_chat_history(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    
    try:
        # SỬA LẠI: order_by('timestamp') cho khớp chuẩn với memory.py
        docs = db.collection('sessions').document(session_id).collection('messages').order_by('timestamp').stream()
        
        messages = []
        for doc in docs:
            msg_data = doc.to_dict()
            
            # Xử lý thời gian từ Firestore sang dạng mili-giây cho React dễ hiểu
            t_val = msg_data.get("timestamp")
            if t_val:
                created_at_ms = int(t_val.timestamp() * 1000)
            else:
                created_at_ms = 0
                
            messages.append({
                "id": doc.id,
                "role": msg_data.get("role", "bot"), 
                "content": msg_data.get("content", ""),
                "createdAt": created_at_ms # React cần trường tên là createdAt
            })
            
        return {"messages": messages}
        
    except Exception as e:
        print(f"Lỗi lấy chi tiết tin nhắn: {e}")
        return {"messages": []}