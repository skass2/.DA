from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth

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
    # Cách đơn giản: Chỉ định cứng các email được phép làm admin
    # Hoặc bạn có thể dùng Firebase Custom Claims: if not current_user.get("admin", False):
    admin_emails = ["admin@example.com"] # <<< THAY ĐỔI EMAIL CỦA BẠN VÀO ĐÂY
    if current_user.get("email") not in admin_emails and not current_user.get("admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Không có quyền truy cập (Chỉ dành cho Admin)"
        )
    return current_user