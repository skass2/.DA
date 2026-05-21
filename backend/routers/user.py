import time

from fastapi import APIRouter, Request, Depends, BackgroundTasks
from rag.pipeline import ask_rag
from rag.memory import get_history, save_message, clear_history, get_db
from rag.loader import load_data
from routers.auth import get_current_user

router = APIRouter(prefix="/user", tags=["User"])

# API Chat chuyển từ main.py sang, có thêm dependency get_current_user
@router.get("/chat")
def chat(request: Request, q: str, background_tasks: BackgroundTasks, session_id: str = "default", current_user: dict = Depends(get_current_user)):
    started_at = time.perf_counter()
    
    if not q or q.strip() == "":
        return {"answer": "Vui lòng nhập câu hỏi."}

    uid = current_user.get('uid')

    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"answer": "Hệ thống đang khởi tạo dữ liệu, vui lòng đợi trong giây lát."}

    history = get_history(session_id)
    answer = ask_rag(db=db, query=q, session_id=session_id, history=history)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    print(f"[CHAT API] uid={uid} session_id={session_id} latency_ms={elapsed_ms}")

    # Đẩy UID và Session ID vào DB dưới dạng Background Task để phản hồi ngay lập tức cho web
    background_tasks.add_task(save_message, uid, session_id, "user", q)
    background_tasks.add_task(save_message, uid, session_id, "bot", answer)

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

# ===== API DÀNH CHO 2 USE CASE XEM VÀ TRA CỨU =====

@router.get("/procedures/search")
def search_procedures(request: Request, q: str):
    """Tra cứu thủ tục theo ngữ nghĩa dựa vào từ khóa hoặc nhu cầu đời thường"""
    db = getattr(request.app.state, "db", None)
    if not db or not q:
        return {"results": []}
        
    # Tìm kiếm ngữ nghĩa trong VectorDB
    docs = db.similarity_search(q, k=20)
    
    matched_procedures = {}
    for d in docs:
        p_id = d.metadata.get("id") or d.metadata.get("procedure_id")
        p_name = d.metadata.get("name")
        if p_id and p_name and p_id not in matched_procedures:
            matched_procedures[p_id] = {
                "id": p_id,
                "name": p_name,
                "linh_vuc": d.metadata.get("linh_vuc", "Chưa phân loại")
            }
            
    return {"results": list(matched_procedures.values())[:10]} # Top 10

@router.get("/procedures/{procedure_id}")
def get_procedure_detail(procedure_id: str):
    """Xem chi tiết thông tin thủ tục: hồ sơ, cơ quan, thời hạn, phí, kết quả, căn cứ pháp lý"""
    data = load_data()
    for item in data:
        if str(item.get("id")) == procedure_id:
            return {"procedure": item}
    return {"error": "Không tìm thấy thủ tục", "procedure": None}