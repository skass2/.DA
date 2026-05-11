import { useEffect, useState } from "react";
import { auth } from "../firebase";
import { onAuthStateChanged } from "firebase/auth";

interface Session {
  session_id: string;
  title: string;
  updated_at: string;
}

interface SidebarProps {
  onSelectSession: (id: string) => void;
  currentSessionId: string;
  isOpen: boolean;
  onToggle: () => void;
}

export default function Sidebar({ onSelectSession, currentSessionId, isOpen, onToggle }: SidebarProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchSessions = async (user: any) => {
      setIsLoading(true);
      try {
        const token = await user.getIdToken();
        const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${apiUrl}/user/sessions`, {
          headers: { 
            "Authorization": `Bearer ${token}`,
            "ngrok-skip-browser-warning": "true"
          }
        });
        if (!res.ok) throw new Error("Lỗi API tải danh sách chat");
        const data = await res.json();
        if (data.sessions) setSessions(data.sessions);
      } catch (error) {
        console.error("Lỗi lấy danh sách chat:", error);
      } finally {
        setIsLoading(false);
      }
    };

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        fetchSessions(user);
      } else {
        setSessions([]);
        setIsLoading(false);
      }
    });
    return () => unsubscribe();
  }, [currentSessionId]);

  return (
    <div className={`absolute md:relative z-50 bg-[#f0f4f9] dark:bg-gray-900 h-full border-gray-200 dark:border-gray-800 flex flex-col shrink-0 transition-all duration-300 overflow-hidden shadow-2xl md:shadow-none ${isOpen ? 'w-72 md:w-64 border-r translate-x-0' : 'w-0 border-r-0 -translate-x-full md:translate-x-0'}`}>
      <div className="w-64 h-full flex flex-col shrink-0">
        <div className="p-3 flex items-center justify-between gap-2">
          <button 
            onClick={() => onSelectSession(`session-${Date.now()}`)} 
            className="flex-1 flex items-center justify-center gap-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-700 font-medium rounded-full px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-500 shadow-sm"
          >
            <svg className="w-5 h-5 text-gray-600 dark:text-gray-400 transition-colors duration-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
            Đoạn chat mới
          </button>
          <button 
            onClick={onToggle}
            className="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors duration-300 shrink-0"
            title="Thu nhỏ thanh bên"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1 mt-2 transition-colors duration-500">
        <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-3 px-2 uppercase tracking-wider transition-colors duration-500">Gần đây</h4>
        
        {isLoading ? (
          <div className="text-gray-400 dark:text-gray-500 text-sm px-2 animate-pulse">Đang tải...</div>
        ) : sessions.length === 0 ? (
          <div className="text-gray-400 dark:text-gray-500 text-sm px-2 italic transition-colors duration-500">Chưa có lịch sử</div>
        ) : (
          sessions.map((s) => (
            <div 
              key={s.session_id} 
              onClick={() => onSelectSession(s.session_id)}
              className={`flex items-center gap-3 cursor-pointer px-3 py-2.5 rounded-lg text-sm transition-all duration-500 ${
                currentSessionId === s.session_id 
                  ? 'bg-[#e1e5ea] dark:bg-gray-700/80 text-blue-800 dark:text-blue-300 font-medium' 
                  : 'text-gray-700 dark:text-gray-300 hover:bg-[#e1e5ea]/60 dark:hover:bg-gray-800/60'
              }`}
            >
              <svg className="w-4 h-4 flex-shrink-0 text-gray-500 dark:text-gray-400 transition-colors duration-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <span className="truncate w-full transition-colors duration-500">{s.title || "Cuộc trò chuyện"}</span>
            </div>
          ))
        )}
      </div>
    </div>
    </div>
  );
}