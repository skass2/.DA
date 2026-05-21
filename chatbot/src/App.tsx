import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import ChatBox from "./components/ChatBox";
import LoginPage from "./components/LoginPage";
import AdminDashboard from "./components/AdminDashboard";
import HomePage from "./pages/HomePage";
import ProcedureDetail from "./pages/ProcedureDetail";
import { auth } from "./firebase";
import { onAuthStateChanged } from "firebase/auth";
import type { User } from "firebase/auth";

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      // KIỂM TRA BẢO MẬT: Nếu có user NHƯNG chưa xác thực email (áp dụng cho tài khoản thường)
      if (currentUser && currentUser.emailVerified === false) {
        setUser(null); // Không cho phép vào app
        setIsAdmin(false);
      } else {
        setUser(currentUser); // Cho phép vào ChatBox (Đã xác thực, hoặc đăng nhập bằng Google)
        if (currentUser) {
          try {
            const token = await currentUser.getIdToken();
            const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(`${apiUrl}/auth/check-admin`, {
              headers: { 
                "Authorization": `Bearer ${token}`,
                "ngrok-skip-browser-warning": "true"
              } 
            });
            const data = await res.json();
            if (data.is_admin) {
              setIsAdmin(true);
            } else {
              setIsAdmin(false);
            }
          } catch (e) { setIsAdmin(false); }
        }
      }
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  if (loading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 transition-colors duration-500">
        <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <div className="h-screen w-screen overflow-hidden">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/procedure/:id" element={<ProcedureDetail />} />
          <Route path="/login" element={user ? <Navigate to="/chat" replace /> : <LoginPage />} />
          <Route path="/chat" element={user ? <ChatBox isAdmin={isAdmin} /> : <Navigate to="/login" replace />} />
          <Route path="/admin" element={user && isAdmin ? <AdminDashboard /> : <Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;