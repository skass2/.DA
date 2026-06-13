import os
import time
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Request, Depends, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from rag.pipeline import ask_rag
from rag.memory import get_history, save_message, clear_history, get_db
from rag.loader import load_data, is_valid_procedure, get_content, get_procedure_name
from routers.auth import get_current_user

router = APIRouter(prefix="/user", tags=["User"])

BACKEND_DIR = Path(__file__).resolve().parents[1]
FORM_DIR = BACKEND_DIR / "data" / "file_Mau"


def make_json_safe(obj):
    """
    Chặn lỗi FastAPI không serialize được numpy.float32/numpy.int64 hoặc ndarray.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return obj.tolist()
        except Exception:
            pass

    return obj


def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def short_code(code: str) -> str:
    parts = clean_text(code).split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return clean_text(code)



def normalize_search_text(value: str) -> str:
    import unicodedata
    value = clean_text(value).lower()
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    return value


def procedure_matches_id(item: dict, procedure_id: str) -> bool:
    content = get_content(item)
    request_id = clean_text(unquote(procedure_id))

    candidates = {
        clean_text(item.get("id")),
        clean_text(item.get("old_internal_id")),
        clean_text(item.get("search_code")),
        clean_text(content.get("Mã thủ tục")),
        short_code(clean_text(item.get("id"))),
        short_code(clean_text(content.get("Mã thủ tục"))),
    }
    return request_id in candidates


# API Chat chuyển từ main.py sang, có thêm dependency get_current_user
@router.get("/chat")
def chat(
    request: Request,
    q: str,
    background_tasks: BackgroundTasks,
    session_id: str = "default",
    current_user: dict = Depends(get_current_user),
):
    started_at = time.perf_counter()

    if not q or q.strip() == "":
        return {"answer": "Vui lòng nhập câu hỏi."}

    uid = current_user.get("uid")

    db = getattr(request.app.state, "db", None)
    if db is None:
        return {"answer": "Hệ thống đang khởi tạo dữ liệu, vui lòng đợi trong giây lát."}

    history = get_history(session_id)
    rag_result = ask_rag(db=db, query=q, session_id=session_id, history=history)

    if isinstance(rag_result, dict):
        answer_text = rag_result.get("answer", "")
    else:
        answer_text = str(rag_result)
        rag_result = {
            "answer": answer_text,
            "suggested_questions": [],
            "selected_procedure": None,
            "sources": [],
            "procedure_candidates": [],
        }

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    print(f"[CHAT API] uid={uid} session_id={session_id} latency_ms={elapsed_ms}")

    background_tasks.add_task(save_message, uid, session_id, "user", q)
    background_tasks.add_task(save_message, uid, session_id, "bot", answer_text)

    return make_json_safe(rag_result)


# ===== API LẤY DANH SÁCH LỊCH SỬ CHAT CHO SIDEBAR =====
@router.get("/sessions")
def get_user_sessions(current_user: dict = Depends(get_current_user)):
    uid = current_user.get("uid")
    db = get_db()

    docs = db.collection("sessions").where("uid", "==", uid)\
        .order_by("updated_at", direction="DESCENDING").stream()

    sessions = []
    for doc in docs:
        data = doc.to_dict()
        sessions.append({
            "session_id": doc.id,
            "title": data.get("title", "Trò chuyện mới"),
            "updated_at": data.get("updated_at"),
        })

    return {"sessions": sessions}


@router.get("/chat/history")
def get_chat_history(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()

    try:
        docs = db.collection("sessions").document(session_id).collection("messages").order_by("timestamp").stream()

        messages = []
        for doc in docs:
            msg_data = doc.to_dict()
            t_val = msg_data.get("timestamp")
            created_at_ms = int(t_val.timestamp() * 1000) if t_val else 0

            messages.append({
                "id": doc.id,
                "role": msg_data.get("role", "bot"),
                "content": msg_data.get("content", ""),
                "createdAt": created_at_ms,
            })

        return {"messages": messages}

    except Exception as e:
        print(f"Lỗi lấy chi tiết tin nhắn: {e}")
        return {"messages": []}


# ===== API DÀNH CHO 2 USE CASE XEM VÀ TRA CỨU =====
@router.get("/procedures/search")
def search_procedures(request: Request, q: str = ""):
    """
    Tra cứu thủ tục theo từ khóa.
    Lọc bỏ bản ghi lỗi/rỗng để HomePage không hiện card trắng.
    """
    data = load_data()
    results = []
    q_norm = normalize_search_text(q)

    for item in data:
        if not is_valid_procedure(item):
            continue

        content = get_content(item)
        name = get_procedure_name(item)
        item_id = clean_text(item.get("id"))
        search_code = clean_text(item.get("search_code") or short_code(content.get("Mã thủ tục") or item_id))
        linh_vuc = clean_text(content.get("Lĩnh vực")) or "Chưa phân loại"
        cap_thuc_hien = clean_text(content.get("Cấp thực hiện"))
        co_quan = clean_text(content.get("Cơ quan thực hiện") or content.get("Cơ quan có thẩm quyền") or content.get("Địa chỉ tiếp nhận HS"))
        doi_tuong = clean_text(content.get("Đối tượng thực hiện"))
        ket_qua = clean_text(content.get("Kết quả thực hiện"))
        file_mau = content.get("File mẫu", [])
        file_mau_count = len(file_mau) if isinstance(file_mau, list) else 0

        searchable = " ".join([
            item_id,
            search_code,
            name,
            linh_vuc,
            cap_thuc_hien,
            co_quan,
            doi_tuong,
            ket_qua,
        ])
        searchable_norm = normalize_search_text(searchable)

        if not q_norm or q_norm in searchable_norm:
            results.append({
                "id": item_id,
                "search_code": search_code,
                "name": name,
                "linh_vuc": linh_vuc,
                "cap_thuc_hien": cap_thuc_hien,
                "co_quan": co_quan,
                "doi_tuong": doi_tuong,
                "ket_qua": ket_qua,
                "file_mau_count": file_mau_count,
            })

    return {"results": results}


@router.get("/procedures/{procedure_id}")
def get_procedure_detail(procedure_id: str):
    """
    Xem chi tiết thông tin thủ tục.
    Nhận cả mã đầy đủ, mã rút gọn và old_internal_id.
    """
    data = load_data()
    for item in data:
        if not is_valid_procedure(item):
            continue
        if procedure_matches_id(item, procedure_id):
            # Đảm bảo tên luôn có ở cấp cao để frontend không bị trắng.
            item = dict(item)
            item["name"] = get_procedure_name(item)
            return {"procedure": item}

    return {"error": "Không tìm thấy thủ tục", "procedure": None}


@router.get("/procedure-forms/{stored_name}")
def get_procedure_form(stored_name: str):
    """
    Tải file mẫu đã cào về từ Cổng DVC.
    Chỉ cho phép lấy file trong backend/data/file_Mau để tránh path traversal.
    """
    decoded_name = clean_text(unquote(stored_name))
    safe_name = Path(decoded_name).name

    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ.")

    file_path = FORM_DIR / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file mẫu trên server.")

    return FileResponse(
        path=str(file_path),
        filename=safe_name,
        media_type="application/octet-stream",
    )
