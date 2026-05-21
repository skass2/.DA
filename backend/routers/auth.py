import os
import random
import string
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from firebase_admin import auth
from rag.memory import get_db

router = APIRouter(prefix="/auth", tags=["Auth"])
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # Xác thực Token được gửi lên từ Frontend thông qua Firebase Admin
        decoded_token = auth.verify_id_token(token)
        return decoded_token  # Trả về dict chứa thông tin user (uid, email, ...)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token không hợp lệ hoặc đã hết hạn: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_admin_user(current_user: dict = Depends(get_current_user)):
    admin_emails = ["ngvinh7021@gmail.com"] # Super Admin mặc định
    try:
        db = get_db()
        doc = db.collection("settings").document("admins").get()
        if doc.exists:
            admin_emails.extend(doc.to_dict().get("emails", []))
    except Exception:
        pass

    if current_user.get("email") not in admin_emails and not current_user.get("admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Không có quyền truy cập (Chỉ dành cho Admin)"
        )
    return current_user

@router.get("/check-admin")
def check_admin(current_user: dict = Depends(get_current_user)):
    try:
        get_admin_user(current_user)
        return {"is_admin": True}
    except:
        return {"is_admin": False}

# ===== DỮ LIỆU ĐẦU VÀO =====
class EmailRequest(BaseModel):
    email: str

class VerifyRequest(BaseModel):
    email: str
    otp_code: str
    password: str
    name: str = ""

# ===== HÀM GỬI EMAIL (Tùy chọn) =====
def send_email_otp(to_email: str, otp_code: str):
    SENDER_EMAIL = os.getenv("SENDER_EMAIL", "your_email@gmail.com")
    SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "your_app_password")
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg['Subject'] = "Mã OTP Đăng Ký Tài Khoản"
    body = f"Chào bạn,\n\nMã OTP xác nhận đăng ký tài khoản của bạn là: {otp_code}\nMã này sẽ hết hạn sau 5 phút.\n\nTrân trọng."
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"[EMAIL ERROR] Không thể gửi email: {e}")

# ===== API GỬI VÀ XÁC MINH OTP =====
@router.post("/send-otp")
def send_otp(req: EmailRequest):
    try:
        auth.get_user_by_email(req.email)
        raise HTTPException(status_code=400, detail="Email này đã được đăng ký!")
    except auth.UserNotFoundError:
        pass 
    otp_code = ''.join(random.choices(string.digits, k=6))
    db = get_db()
    db.collection("otp_codes").document(req.email).set({
        "code": otp_code,
        "expires_at": datetime.now() + timedelta(minutes=5)
    })
    print(f"MÃ OTP CHO {req.email} LÀ: {otp_code}") 
    # send_email_otp(req.email, otp_code) # Bỏ comment dòng này khi bạn đã cấu hình Email
    return {"message": "Mã xác nhận đã được gửi đến email của bạn."}

@router.post("/verify-otp")
def verify_otp_and_register(req: VerifyRequest):
    db = get_db()
    doc_ref = db.collection("otp_codes").document(req.email)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=400, detail="Không tìm thấy yêu cầu gửi mã cho email này.")
    data = doc.to_dict()
    if datetime.now().timestamp() > data["expires_at"].timestamp():
        raise HTTPException(status_code=400, detail="Mã OTP đã hết hạn.")
    if data["code"] != req.otp_code:
        raise HTTPException(status_code=400, detail="Mã OTP không chính xác.")
    try:
        user_record = auth.create_user(email=req.email, password=req.password, display_name=req.name, email_verified=True)
        doc_ref.delete()
        return {"message": "Đăng ký thành công!", "uid": user_record.uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi tạo tài khoản: {str(e)}")