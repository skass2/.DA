import { useState, useEffect } from "react";
import Message from "./Message";
import InputBox from "./InputBox";
import Sidebar from "./Sidebar";
import type { ChatMessage } from "../types/chat";
import { auth } from "../firebase";
import { onAuthStateChanged, type User as FirebaseUser } from "firebase/auth";
import { useLocation, useNavigate } from "react-router-dom";
import UserMenu from "./UserMenu";

interface ChatBoxProps {
  isAdmin?: boolean;
}

export default function ChatBox({ isAdmin }: ChatBoxProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [firebaseUser, setFirebaseUser] = useState<FirebaseUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>(`session-${Date.now()}`);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      setFirebaseUser(currentUser);
      if (currentUser) {
        const idToken = await currentUser.getIdToken();
        setToken(idToken);
      } else {
        setToken(null);
      }
    });
    return () => unsubscribe();
  }, []);

  // Bắt sự kiện chuyển hướng từ trang chi tiết thủ tục
  useEffect(() => {
    if (location.state && location.state.anchorMessage && token) {
      const initMessage = location.state.anchorMessage;
      sendMessage(initMessage);
      // Xóa state để tránh gửi lại tin nhắn khi người dùng reload trang
      navigate(location.pathname, { replace: true, state: {} });
    }
  }, [location.state, token]);

  // Lắng nghe sự kiện đổi tên từ UserMenu để cập nhật lại lời chào
  useEffect(() => {
    const handleProfileUpdate = () => {
      if (auth.currentUser) setFirebaseUser({ ...auth.currentUser } as FirebaseUser);
    };
    window.addEventListener('profileUpdated', handleProfileUpdate);
    return () => window.removeEventListener('profileUpdated', handleProfileUpdate);
  }, []);

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    if (newMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const sendMessage = async (text: string) => {
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    if (!firebaseUser || !token) {
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), role: "bot", content: "Lỗi xác thực, vui lòng tải lại trang.", createdAt: Date.now() },
      ]);
      setLoading(false);
      return;
    }

    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(
        `${apiUrl}/user/chat?q=${encodeURIComponent(text)}&session_id=${sessionId}`,
        {
          headers: {
            "Authorization": `Bearer ${token}`,
            "ngrok-skip-browser-warning": "true"
          }
        }
      );

      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      const data = await res.json();

      const botMsg: ChatMessage = {
        id: Date.now().toString(),
        role: "bot",
        content: data.answer || "Không có phản hồi",
        createdAt: Date.now(),
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error("FETCH ERROR:", err);
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), role: "bot", content: "Lỗi kết nối server", createdAt: Date.now() },
      ]);
    }
    setLoading(false);
  };

  return (
    <div className="flex h-full w-full overflow-hidden transition-colors duration-500">
      {/* Sidebar */}
      {firebaseUser && (
        <Sidebar 
          isOpen={isSidebarOpen}
          onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
          currentSessionId={sessionId} 
          onSelectSession={async (id) => {
            setSessionId(id);
            setMessages([]);
            setLoading(true);
            try {
              const currentToken = await firebaseUser.getIdToken();
              const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
              const res = await fetch(`${apiUrl}/user/chat/history?session_id=${id}`, {
                headers: {
                  "Authorization": `Bearer ${currentToken}`,
                  "ngrok-skip-browser-warning": "true"
                }
              });
              if (res.ok) {
                const data = await res.json();
                if (data.messages && data.messages.length > 0) {
                  setMessages(data.messages);
                }
              }
            } catch (error) {
              console.error("Lỗi khi tải lịch sử:", error);
            } finally {
              setLoading(false);
            }
          }} 
        />
      )}

      {/* Lớp phủ cho Mobile khi mở Sidebar */}
      {isSidebarOpen && firebaseUser && (
        <div 
          className="fixed inset-0 bg-black/40 z-40 md:hidden" 
          onClick={() => setIsSidebarOpen(false)}
        ></div>
      )}

      {/* Main Chat Area */}
      <div className="flex flex-col flex-1 h-full bg-[#f2f6fc] dark:bg-gray-900 transition-colors duration-500 relative">
        
        {/* HEADER */}
        <div className="relative z-20 flex justify-between items-center p-2 sm:p-4 border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-md shadow-sm shrink-0 transition-colors duration-500 gap-2">
          <div className="flex items-center gap-1 sm:gap-3 shrink-0">
            {firebaseUser && !isSidebarOpen && (
              <button 
                onClick={() => setIsSidebarOpen(true)}
                className="p-1.5 sm:p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-colors duration-300"
                title="Mở thanh bên"
              >
                <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
              </button>
            )}
            <button 
              onClick={() => navigate("/")}
              className="p-1.5 sm:p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-colors duration-300 flex items-center gap-1"
              title="Về Trang chủ"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>
              <span className="hidden lg:inline text-sm font-medium">Trang chủ</span>
            </button>
            <div className="hidden sm:block">
              <h3 className="text-base sm:text-xl font-bold text-blue-600 dark:text-blue-400 transition-colors duration-500 whitespace-nowrap">Chatbot Thủ Tục</h3>
              <p className="text-[10px] uppercase tracking-widest text-gray-400 dark:text-gray-500 font-medium transition-colors duration-500 hidden md:block">
                Sẵn sàng hỗ trợ bạn
              </p>
            </div>
          </div>
          
          <div className="flex gap-2 sm:gap-4 items-center shrink-1 justify-end">
            {/* Thanh tìm kiếm */}
            <div className="hidden xl:block">
              <input 
                type="text"
                placeholder="Tra cứu thủ tục..."
                className="px-4 py-1.5 text-sm rounded-full border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 w-48 lg:w-64 transition-all duration-300"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                    navigate(`/?search=${encodeURIComponent(e.currentTarget.value.trim())}`);
                  }
                }}
              />
            </div>
            
            <div className="flex items-center gap-1 sm:gap-2 shrink-0">
              {isAdmin && (
                <button onClick={() => navigate("/admin")} className="px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-green-500 text-white font-medium hover:bg-green-600 transition-all duration-500 shadow-md text-xs sm:text-sm whitespace-nowrap">
                  Quản trị
                </button>
              )}
              <button
                onClick={toggleDarkMode}
                className="p-1.5 sm:p-2 rounded-full border border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-all duration-500"
                title={darkMode ? "Bật chế độ sáng" : "Bật chế độ tối"}
              >
                {darkMode ? "☀️" : "🌙"}
              </button>
            </div>
            <UserMenu />
          </div>
        </div>

        {/* MESSAGES */}
        <div className="relative z-10 flex-1 overflow-y-auto p-6 transition-colors duration-500">
          {messages.length > 0 ? (
            messages.map((m) => (
              <Message key={m.id} message={m} />
            ))
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-4 transition-opacity duration-500">
              <div className="text-center space-y-4 max-w-md">
                <h2 className="text-2xl font-semibold text-gray-700 dark:text-gray-300 transition-colors duration-500">
                  Xin chào, {firebaseUser?.displayName || "bạn"}!
                </h2>
                <p className="text-gray-500 dark:text-gray-400 transition-colors duration-500">
                  Tôi có thể giúp gì cho bạn về các thủ tục hành chính hôm nay?
                </p>
              </div>
            </div>
          )}
          {loading && (
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-sm mt-2 transition-colors duration-500">
              <span className="flex space-x-1">
                <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"></span>
                <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce delay-150"></span>
                <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce delay-300"></span>
              </span>
              <span>Bot đang gõ...</span>
            </div>
          )}
        </div>

        {/* INPUT */}
        <div className="relative z-10 p-4 border-t bg-white dark:bg-gray-800 shrink-0 transition-colors duration-500">
          <InputBox onSend={sendMessage} loading={loading} />
        </div>

      </div>
    </div>
  );
}