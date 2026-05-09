import { useState, useEffect } from "react";
import Message from "./Message";
import InputBox from "./InputBox";
import Sidebar from "./Sidebar";
import type { ChatMessage } from "../types/chat";
import { auth, signOut } from "../firebase";
import { onAuthStateChanged, type User as FirebaseUser } from "firebase/auth";

export default function ChatBox() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [firebaseUser, setFirebaseUser] = useState<FirebaseUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>(`session-${Date.now()}`);

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

  const handleLogout = async () => {
    await signOut(auth);
  };

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

      {/* Main Chat Area */}
      <div className="flex flex-col flex-1 h-full bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-900 dark:to-gray-800 transition-colors duration-500">
        
        {/* HEADER */}
        <div className="flex justify-between items-center p-4 border-b bg-white/70 dark:bg-gray-800/70 backdrop-blur-md shadow-md shrink-0 transition-colors duration-500">
          <div>
            <h3 className="text-xl font-bold text-blue-600 dark:text-blue-400 transition-colors duration-500">Chatbot Thủ Tục</h3>
            <p className="text-[10px] uppercase tracking-widest text-gray-400 dark:text-gray-500 font-medium transition-colors duration-500">
              Sẵn sàng hỗ trợ bạn
            </p>
          </div>
          
          <div className="flex gap-4 items-center">
            {firebaseUser && (
              <div className="flex items-center gap-3 bg-gray-100 dark:bg-gray-700 px-3 py-1.5 rounded-full border border-gray-200 dark:border-gray-600 shadow-sm transition-colors duration-500">
                {firebaseUser.photoURL ? (
                  <img src={firebaseUser.photoURL} alt="Avatar" className="w-7 h-7 rounded-full shadow-sm" />
                ) : (
                  <div className="w-7 h-7 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                    {firebaseUser.displayName?.charAt(0).toUpperCase() || firebaseUser.email?.charAt(0).toUpperCase()}
                  </div>
                )}
                <span className="text-gray-700 dark:text-gray-200 font-medium text-sm hidden sm:inline-block transition-colors duration-500">
                  {firebaseUser.displayName || firebaseUser.email}
                </span>
              </div>
            )}
            
            <div className="flex items-center gap-2">
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-full border border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-all duration-500"
                title={darkMode ? "Bật chế độ sáng" : "Bật chế độ tối"}
              >
                {darkMode ? "☀️" : "🌙"}
              </button>
              <button 
                onClick={handleLogout} 
                className="px-4 py-1.5 rounded bg-red-500 text-white font-medium hover:bg-red-600 transition-all duration-500 shadow-md text-sm"
              >
                Đăng xuất
              </button>
            </div>
          </div>
        </div>

        {/* MESSAGES */}
        <div className="flex-1 overflow-y-auto p-6 transition-colors duration-500 relative">
          {messages.length > 0 ? (
            messages.map((m) => (
              <Message key={m.id} message={m} />
            ))
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-4 transition-opacity duration-1000">
              <div className="text-center space-y-4 max-w-md">
                <h2 className="text-2xl font-semibold text-gray-700 dark:text-gray-300 transition-colors duration-500">
                  Xin chào, {firebaseUser?.displayName || "bạn"}!
                </h2>
                <p className="text-gray-500 dark:text-gray-400 transition-colors duration-500">
                  Tôi có thể giúp gì cho bạn về các thủ tục hành chính hôm nay?
                </p>
                <div className="grid grid-cols-1 gap-2 pt-4">
                  {[
                    "Thủ tục làm hộ chiếu cần những gì?",
                    "Hướng dẫn đăng ký tạm trú trực tuyến",
                    "Cách tra cứu mã số thuế cá nhân"
                  ].map((hint, i) => (
                    <button
                      key={i}
                      onClick={() => sendMessage(hint)}
                      className="text-sm px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-xl text-gray-600 dark:text-gray-400 hover:bg-white dark:hover:bg-gray-700 transition-all duration-500 shadow-sm"
                    >
                      "{hint}"
                    </button>
                  ))}
                </div>
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
        <div className="p-4 border-t bg-white dark:bg-gray-800 shrink-0 transition-colors duration-500">
          <InputBox onSend={sendMessage} loading={loading} />
        </div>
      </div>
    </div>
  );
}