from fastapi import APIRouter, Request, Depends
from routers.auth import get_admin_user
from rag.loader import load_data
from rag.chunker import create_chunks
from rag.vectorstore import build_vectorstore
from rag.memory import get_db, clear_history
from rag.pipeline import smart_llm_invoke
from firebase_admin import auth
import datetime
import json
import os
import shutil
import time
from pydantic import BaseModel

router = APIRouter(prefix="/admin", tags=["Admin"])

# API Admin: Khởi tạo lại ChromaDB nếu file JSON bị thay đổi
@router.post("/reload-vectordb")
def reload_vectordb(request: Request, current_admin: dict = Depends(get_admin_user)):
    try:
        data = load_data()
        chunks = create_chunks(data)
        
        # Quan trọng: Giải phóng DB hiện tại khỏi RAM để hệ điều hành nhả file lock
        request.app.state.db = None
        
        request.app.state.db = build_vectorstore(chunks, backup=True)
        
        # Tự động dọn dẹp backup cũ dựa trên cài đặt
        db = get_db()
        doc = db.collection("settings").document("backup").get()
        retention_days = doc.to_dict().get("retention_days", 3) if doc.exists else 3
        
        if retention_days > 0:
            current_time = time.time()
            for item in os.listdir("."):
                if item.startswith("chroma_db_backup_") and os.path.isdir(item):
                    folder_time = os.path.getmtime(item)
                    if (current_time - folder_time) > retention_days * 86400:
                        try:
                            shutil.rmtree(item)
                            print(f"Đã tự động xóa backup quá hạn: {item}")
                        except Exception as e:
                            print(f"Lỗi xóa backup {item}: {e}")
                            
        return {"status": "success", "message": "Cơ sở dữ liệu Vector đã được cập nhật thành công. Đã tự động tạo bản sao lưu (Backup) cho dữ liệu cũ!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 1. Thống kê hệ thống
@router.get("/stats")
def get_system_stats(current_admin: dict = Depends(get_admin_user)):
    db = get_db()
    
    # Lấy tất cả các phiên chat để thống kê
    sessions_ref = db.collection('sessions').stream()
    total_sessions = 0
    
    # Khởi tạo mảng thống kê 7 ngày gần nhất
    today = datetime.datetime.now(datetime.timezone.utc).date()
    chart_data_dict = {}
    for i in range(6, -1, -1):
        day_str = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        chart_data_dict[day_str] = 0
        
    for doc in sessions_ref:
        total_sessions += 1
        data = doc.to_dict()
        created_at = data.get("created_at")
        if created_at:
            try:
                # Convert Firestore Datetime sang chuỗi Ngày
                dt_str = created_at.date().strftime("%Y-%m-%d")
                if dt_str in chart_data_dict:
                    chart_data_dict[dt_str] += 1
            except:
                pass
                
    # Đưa data vào mảng cho Recharts
    chart_data = [{"date": k, "sessions": v} for k, v in chart_data_dict.items()]
    
    # Lấy danh sách user từ Firebase Auth
    try:
        users = auth.list_users(max_results=1000)
        total_users = len(users.users)
    except Exception as e:
        total_users = 0
        
    return {
        "total_users": total_users, 
        "total_sessions": total_sessions, 
        "status": "Healthy",
        "chart_data": chart_data
    }

# 2. Giám sát lịch sử: Lấy danh sách Người dùng
@router.get("/users")
def get_all_users(current_admin: dict = Depends(get_admin_user)):
    try:
        users_page = auth.list_users(max_results=1000)
        user_list = []
        for u in users_page.users:
            user_list.append({
                "uid": u.uid,
                "email": u.email,
                "displayName": u.display_name or "Người dùng ẩn danh",
                "photoURL": u.photo_url or ""
            })
        return {"users": user_list}
    except Exception as e:
        return {"users": [], "error": str(e)}

# Lấy danh sách phiên chat theo UID của User
@router.get("/users/{uid}/sessions")
def get_user_sessions_admin(uid: str, current_admin: dict = Depends(get_admin_user)):
    db = get_db()
    docs = db.collection('sessions').where('uid', '==', uid).order_by('updated_at', direction='DESCENDING').stream()
    sessions = [{"session_id": doc.id, "title": doc.to_dict().get("title", "Trò chuyện mới")} for doc in docs]
    return {"sessions": sessions}

# Lấy chi tiết tin nhắn của 1 phiên chat bất kỳ (Dành cho Admin)
@router.get("/history/sessions/{session_id}")
def get_admin_chat_history(session_id: str, current_admin: dict = Depends(get_admin_user)):
    db = get_db()
    docs = db.collection('sessions').document(session_id).collection('messages').order_by('timestamp').stream()
    messages = [{"id": doc.id, "role": doc.to_dict().get("role"), "content": doc.to_dict().get("content")} for doc in docs]
    
    return {"messages": messages}

# Xóa phiên chat bất kỳ
@router.delete("/history/sessions/{session_id}")
def delete_admin_chat_history(session_id: str, current_admin: dict = Depends(get_admin_user)):
    try:
        clear_history(session_id)
        return {"status": "success", "message": "Đã xóa phiên chat thành công"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ===== API QUẢN LÝ BACKUP =====
class BackupSettingModel(BaseModel):
    retention_days: int

@router.get("/backups")
def get_backups(current_admin: dict = Depends(get_admin_user)):
    backups = []
    try:
        for item in os.listdir("."):
            if item.startswith("chroma_db_backup_") and os.path.isdir(item):
                size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(item) for filename in filenames)
                backups.append({
                    "name": item,
                    "size": f"{size / (1024*1024):.2f} MB",
                    "mtime": os.path.getmtime(item)
                })
        backups.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception as e:
        pass
    return {"backups": backups}

@router.delete("/backups/{backup_name}")
def delete_backup(backup_name: str, current_admin: dict = Depends(get_admin_user)):
    if not backup_name.startswith("chroma_db_backup_") or ".." in backup_name:
        return {"status": "error", "message": "Tên backup không hợp lệ"}
    try:
        if os.path.exists(backup_name):
            shutil.rmtree(backup_name)
            return {"status": "success", "message": "Xóa backup thành công"}
        return {"status": "error", "message": "Không tìm thấy thư mục backup"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/backups/settings")
def save_backup_settings(data: BackupSettingModel, current_admin: dict = Depends(get_admin_user)):
    db = get_db()
    db.collection("settings").document("backup").set({"retention_days": data.retention_days})
    return {"status": "success", "message": "Lưu cài đặt thành công"}

@router.get("/backups/settings")
def get_backup_settings(current_admin: dict = Depends(get_admin_user)):
    db = get_db()
    doc = db.collection("settings").document("backup").get()
    if doc.exists:
        return {"retention_days": doc.to_dict().get("retention_days", 3)}
    return {"retention_days": 3}

# ===== API QUẢN LÝ ADMIN =====
class AdminEmailModel(BaseModel):
    email: str

@router.get("/admins")
def get_admins_list(current_admin: dict = Depends(get_admin_user)):
    db = get_db()
    doc = db.collection("settings").document("admins").get()
    emails = doc.to_dict().get("emails", []) if doc.exists else []
    return {"admins": emails}

@router.post("/admins")
def add_admin_email(data: AdminEmailModel, current_admin: dict = Depends(get_admin_user)):
    db = get_db()
    doc_ref = db.collection("settings").document("admins")
    emails = doc_ref.get().to_dict().get("emails", []) if doc_ref.get().exists else []
    email_to_add = data.email.strip().lower()
    if email_to_add not in emails and email_to_add != "ngvinh7021@gmail.com":
        emails.append(email_to_add)
        doc_ref.set({"emails": emails}, merge=True)
    return {"status": "success", "message": f"Đã cấp quyền Admin cho {email_to_add}"}

@router.delete("/admins/{email}")
def remove_admin_email(email: str, current_admin: dict = Depends(get_admin_user)):
    if email == "ngvinh7021@gmail.com":
        return {"status": "error", "message": "Không thể xóa Super Admin"}
    db = get_db()
    doc_ref = db.collection("settings").document("admins")
    emails = doc_ref.get().to_dict().get("emails", []) if doc_ref.get().exists else []
    if email in emails:
        emails.remove(email)
        doc_ref.set({"emails": emails}, merge=True)
    return {"status": "success", "message": f"Đã thu hồi quyền Admin của {email}"}

# 3. Quản lý tài liệu pháp luật (Lấy danh sách từ procedures.json)
@router.get("/documents")
def get_documents_list(current_admin: dict = Depends(get_admin_user)):
    try:
        data = load_data()
        docs = []
        for item in data:
            docs.append({
                "id": item.get("id", ""),
                "name": item.get("name", "Không xác định"),
                "link": item.get("link", "#"),
                "linh_vuc": item.get("content", {}).get("Lĩnh vực", "Chưa phân loại"),
                "content": item.get("content", {})
            })
        return {"documents": docs}
    except Exception as e:
        return {"documents": [], "error": str(e)}

# ===== API CRUD TÀI LIỆU =====
class ProcedureModel(BaseModel):
    id: str
    name: str
    link: str
    content: dict

# Khắc phục lỗi Sửa/Xóa bằng đường dẫn tuyệt đối
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCEDURES_FILE = os.path.join(BASE_DIR, "data", "procedures.json")

class NLConvertRequest(BaseModel):
    text: str

@router.post("/convert-json")
def convert_nl_to_json(req: NLConvertRequest, current_admin: dict = Depends(get_admin_user)):
    try:
        prompt = f"""Bạn là một trợ lý AI chuyên xử lý dữ liệu thủ tục hành chính.
Hãy đọc văn bản dưới đây và trích xuất các thông tin thành một chuỗi JSON hợp lệ.
Các trường dữ liệu cần trích xuất (nếu không có thông tin thì để chuỗi rỗng ""):
"Lĩnh vực", "Trình tự thực hiện", "Cách thức thực hiện", "Thành phần hồ sơ", "Thời hạn giải quyết", "Cơ quan thực hiện", "Phí", "Lệ phí", "Yêu cầu điều kiện", "Căn cứ pháp lý".

Chỉ trả về ĐÚNG MỘT CHUỖI JSON, KHÔNG BÌNH LUẬN GÌ THÊM. KHÔNG dùng markdown ```json.

Văn bản gốc:
{req.text}
"""
        res = smart_llm_invoke(prompt)
        if res:
            res = res.strip()
            if res.startswith("```json"):
                res = res[7:]
            if res.startswith("```"):
                res = res[3:]
            if res.endswith("```"):
                res = res[:-3]
            res = res.strip()
            return {"status": "success", "data": res}
        return {"status": "error", "message": "Không nhận được phản hồi từ AI"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _load_json():
    if not os.path.exists(PROCEDURES_FILE):
        return []
    with open(PROCEDURES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(data):
    with open(PROCEDURES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

@router.post("/documents")
def create_document(doc: ProcedureModel, current_admin: dict = Depends(get_admin_user)):
    try:
        data = _load_json()
        new_doc = {"id": doc.id, "name": doc.name, "link": doc.link, "content": doc.content}
        data.append(new_doc)
        _save_json(data)
        return {"status": "success", "message": "Thêm tài liệu thành công"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.put("/documents/{doc_id}")
def update_document(doc_id: str, doc: ProcedureModel, current_admin: dict = Depends(get_admin_user)):
    try:
        data = _load_json()
        for i, item in enumerate(data):
            if str(item.get("id")) == str(doc_id):
                data[i] = {"id": doc.id, "name": doc.name, "link": doc.link, "content": doc.content}
                _save_json(data)
                return {"status": "success", "message": "Cập nhật tài liệu thành công"}
        return {"status": "error", "message": "Không tìm thấy tài liệu"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, current_admin: dict = Depends(get_admin_user)):
    try:
        data = _load_json()
        new_data = [item for item in data if str(item.get("id")) != str(doc_id)]
        if len(data) == len(new_data):
            return {"status": "error", "message": "Không tìm thấy tài liệu"}
        _save_json(new_data)
        return {"status": "success", "message": "Xóa tài liệu thành công"}
    except Exception as e:
        return {"status": "error", "message": str(e)}