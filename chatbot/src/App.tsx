import { useEffect, useState } from "react";
import ChatBox from "./components/ChatBox";
import LoginPage from "./components/LoginPage";
import AdminDashboard from "./components/AdminDashboard";
import { auth } from "./firebase";
import { onAuthStateChanged } from "firebase/auth";
import type { User } from "firebase/auth";

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);
  const [viewMode, setViewMode] = useState<"admin" | "user">("user");

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
            const res = await fetch(`${apiUrl}/auth/check-admin`, {
              headers: { 
                "Authorization": `Bearer ${token}`,
                "ngrok-skip-browser-warning": "true"
              } 
            });
            const data = await res.json();
            if (data.is_admin) {
              setIsAdmin(true);
              setViewMode("admin");
            } else {
              setIsAdmin(false);
              setViewMode("user");
            }
          } catch (e) { setIsAdmin(false); setViewMode("user"); }
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
    <div className="h-screen w-screen overflow-hidden">
      {user 
        ? (viewMode === "admin" 
            ? <AdminDashboard onSwitchMode={() => setViewMode("user")} /> 
            : <ChatBox isAdmin={isAdmin} onSwitchMode={() => setViewMode("admin")} />) 
        : <LoginPage />}
    </div>
  );
}

export default App;