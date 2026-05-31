import React, { useState } from "react";
import { auth, googleProvider, signInWithPopup, signOut } from "../firebase";
import { 
  signInWithEmailAndPassword
} from "firebase/auth";

export default function LoginPage() {
  const [isRegistering, setIsRegistering] = useState(false);
  const [view, setView] = useState<'form' | 'otp'>('form');
  
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [otpCode, setOtpCode] = useState("");
  
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [error, setError] = useState("");
  const [successMsg, setSuccessMsg] = useState("");
  const [loading, setLoading] = useState(false);

  // LOGIC KIỂM TRA MẬT KHẨU MẠNH
  const isLengthValid = password.length >= 8;
  const hasUpperCase = /[A-Z]/.test(password);
  const hasLowerCase = /[a-z]/.test(password);
  const hasNumber = /[0-9]/.test(password);
  const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
  const isPasswordValid = isLengthValid && hasUpperCase && hasLowerCase && hasNumber && hasSpecialChar;

  const handleStandardAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccessMsg("");
    setLoading(true);

    try {
      if (isRegistering) {
        if (!isPasswordValid) {
          setError("Vui lòng đáp ứng tất cả các yêu cầu về bảo mật mật khẩu bên dưới!");
          setLoading(false);
          return;
        }
        if (password !== confirmPassword) {
          setError("Mật khẩu xác nhận không khớp!");
          setLoading(false);
          return;
        }

        // BƯỚC 1: GỌI API BACKEND ĐỂ GỬI MÃ OTP
        const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${apiUrl}/auth/send-otp`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email })
        });
        const data = await res.json();
        
        if (!res.ok) {
          throw new Error(data.detail || "Lỗi khi gửi OTP!");
        }
        
        setView('otp');
        setSuccessMsg(data.message || `Mã OTP đã được gửi tới email ${email}`);
        
      } else {
        // ĐĂNG NHẬP
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        
        // KIỂM TRA NẾU CHƯA XÁC THỰC MAIL THÌ ĐÁ RA NGOÀI
        if (!userCredential.user.emailVerified) {
          await signOut(auth);
          setError("Tài khoản này chưa được xác thực. Vui lòng kiểm tra email của bạn để kích hoạt!");
        }
      }
    } catch (err: any) {
      console.error(err);
      if (err.code === 'auth/email-already-in-use') setError("Email này đã được sử dụng!");
      else if (err.code === 'auth/wrong-password' || err.code === 'auth/user-not-found' || err.code === 'auth/invalid-credential') setError("Email hoặc mật khẩu không đúng!");
      else setError("Lỗi hệ thống: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiUrl}/auth/verify-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, otp_code: otpCode, password, name })
      });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail || "Xác thực OTP thất bại");

      // BƯỚC 3: ĐĂNG NHẬP VÀO FIREBASE SAU KHI TẠO TÀI KHOẢN THÀNH CÔNG Ở BACKEND
      await signInWithEmailAndPassword(auth, email, password);
      // Lúc này Firebase state sẽ tự cập nhật và đá user vào trang trong.
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setError("");
    setSuccessMsg("");
    try {
      // Google auto-verify email nên không cần hàm chặn
      await signInWithPopup(auth, googleProvider);
    } catch (err: any) {
      setError("Lỗi đăng nhập Google: " + err.message);
    }
  };

  const toggleMode = () => {
    setIsRegistering(!isRegistering);
    setError(""); 
    setSuccessMsg("");
    setPassword("");
    setConfirmPassword("");
    setShowPassword(false);
    setShowConfirmPassword(false);
    setView('form');
    setOtpCode("");
  };

  const EyeIcon = (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
  );

  const EyeSlashIcon = (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" /></svg>
  );

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 dark:bg-gray-900 transition-colors duration-500 p-4">
      <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg w-full max-w-md border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-center text-gray-800 dark:text-white mb-6">
          {view === 'otp' ? "Xác thực OTP" : (isRegistering ? "Tạo tài khoản" : "Đăng nhập")}
        </h2>
        
        {error && (
          <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg text-sm text-center font-medium">
            {error}
          </div>
        )}

        {successMsg && (
          <div className="mb-4 p-3 bg-green-100 border border-green-400 text-green-800 rounded-lg text-sm text-center font-medium">
            {successMsg}
          </div>
        )}

        {view === 'form' ? (
          <form onSubmit={handleStandardAuth} className="space-y-4">
            {isRegistering && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Tên hiển thị</label>
              <input 
                type="text" 
                className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900 dark:text-white"
                placeholder="VD: Nguyễn Văn A"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required={isRegistering}
              />
            </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Email</label>
              <input 
                type="email" 
                className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900 dark:text-white"
                placeholder="Nhập email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Mật khẩu</label>
              <div className="relative">
                <input 
                  type={showPassword ? "text" : "password"} 
                  className="w-full px-4 py-2 pr-10 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900 dark:text-white"
                  placeholder="Nhập mật khẩu"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  {showPassword ? EyeSlashIcon : EyeIcon}
                </button>
              </div>

              {/* HIỂN THỊ LUẬT MẬT KHẨU KHI ĐĂNG KÝ */}
              {isRegistering && password.length > 0 && (
                <div className="mt-2 grid grid-cols-2 gap-1 text-xs font-medium">
                  <span className={isLengthValid ? "text-green-600 dark:text-green-400" : "text-gray-500 dark:text-gray-400"}>
                    {isLengthValid ? "✓" : "○"} Tối thiểu 8 ký tự
                  </span>
                  <span className={hasUpperCase ? "text-green-600 dark:text-green-400" : "text-gray-500 dark:text-gray-400"}>
                    {hasUpperCase ? "✓" : "○"} Ít nhất 1 chữ hoa
                  </span>
                  <span className={hasLowerCase ? "text-green-600 dark:text-green-400" : "text-gray-500 dark:text-gray-400"}>
                    {hasLowerCase ? "✓" : "○"} Ít nhất 1 chữ thường
                  </span>
                  <span className={hasNumber ? "text-green-600 dark:text-green-400" : "text-gray-500 dark:text-gray-400"}>
                    {hasNumber ? "✓" : "○"} Ít nhất 1 chữ số
                  </span>
                  <span className={`col-span-2 ${hasSpecialChar ? "text-green-600 dark:text-green-400" : "text-gray-500 dark:text-gray-400"}`}>
                    {hasSpecialChar ? "✓" : "○"} Ít nhất 1 ký tự đặc biệt (!@#$%^&*)
                  </span>
                </div>
              )}
            </div>

            {isRegistering && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Xác nhận mật khẩu</label>
                <div className="relative">
                  <input 
                    type={showConfirmPassword ? "text" : "password"} 
                    className="w-full px-4 py-2 pr-10 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900 dark:text-white"
                    placeholder="Nhập lại mật khẩu"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required={isRegistering}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    {showConfirmPassword ? EyeSlashIcon : EyeIcon}
                  </button>
                </div>
              </div>
            )}

            <button 
              type="submit" 
              disabled={loading}
              className="w-full bg-blue-600 text-white font-semibold py-2.5 rounded-lg hover:bg-blue-700 transition duration-300 disabled:opacity-60 shadow-md flex justify-center items-center gap-2"
            >
              {loading && (
                <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
              )}
              {loading ? "Đang xử lý..." : (isRegistering ? "Đăng ký" : "Đăng nhập")}
            </button>
          </form>
        ) : (
          <form onSubmit={handleVerifyOtp} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Mã OTP (6 chữ số)</label>
              <input 
                type="text" 
                maxLength={6}
                className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-900 dark:text-white text-center text-2xl tracking-[0.5em] font-bold"
                placeholder="------"
                value={otpCode}
                onChange={(e) => setOtpCode(e.target.value.replace(/[^0-9]/g, ''))}
                required
              />
            </div>
            <button 
              type="submit" 
              disabled={loading || otpCode.length !== 6}
              className="w-full bg-green-600 text-white font-semibold py-2.5 rounded-lg hover:bg-green-700 transition duration-300 disabled:opacity-60 shadow-md flex justify-center items-center gap-2"
            >
              {loading ? "Đang xác thực..." : "Xác nhận OTP"}
            </button>
            <button 
              type="button" 
              onClick={() => setView('form')}
              className="w-full bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-white font-semibold py-2.5 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-500 transition duration-300 shadow-md"
            >
              Quay lại
            </button>
          </form>
        )}

        {view === 'form' && (
          <>
            <div className="mt-5 text-center">
              <button 
                type="button" 
                onClick={toggleMode}
                className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 hover:underline"
              >
                {isRegistering ? "Đã có tài khoản? Đăng nhập tại đây" : "Chưa có tài khoản? Đăng ký ngay"}
              </button>
            </div>

            <div className="mt-7 flex items-center justify-center space-x-3">
              <div className="h-px bg-gray-300 dark:bg-gray-600 flex-1"></div>
              <span className="text-gray-500 dark:text-gray-400 text-sm font-medium px-2">Hoặc tiếp tục với</span>
              <div className="h-px bg-gray-300 dark:bg-gray-600 flex-1"></div>
            </div>

            <button 
              onClick={handleGoogleLogin}
              type="button"
              className="mt-6 w-full flex items-center justify-center gap-3 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-700 dark:text-white font-medium py-2.5 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition shadow-sm"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
              </svg>
              Google
            </button>
          </>
        )}
      </div>
    </div>
  );
}