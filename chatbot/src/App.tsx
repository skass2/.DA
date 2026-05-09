import { useEffect, useState } from "react";
import ChatBox from "./components/ChatBox";
import LoginPage from "./components/LoginPage";
import { auth } from "./firebase";
import { onAuthStateChanged } from "firebase/auth";
import type { User } from "firebase/auth";

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      // KIỂM TRA BẢO MẬT: Nếu có user NHƯNG chưa xác thực email (áp dụng cho tài khoản thường)
      if (currentUser && currentUser.emailVerified === false) {
        setUser(null); // Không cho phép vào app
      } else {
        setUser(currentUser); // Cho phép vào ChatBox (Đã xác thực, hoặc đăng nhập bằng Google)
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
    <div className="h-screen w-screen">
      {user ? <ChatBox /> : <LoginPage />}
    </div>
  );
}

export default App;