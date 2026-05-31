import React, { useState, useEffect } from "react";
import { auth } from "../firebase";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useNavigate } from "react-router-dom";
import UserMenu from "./UserMenu";

export default function AdminDashboard() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<"stats" | "history" | "documents" | "admins">("stats");
  const [stats, setStats] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [sessionMessages, setSessionMessages] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<any[]>([]);
  const [users, setUsers] = useState<any[]>([]);
  const [selectedUid, setSelectedUid] = useState<string | null>(null);
  const [backups, setBackups] = useState<any[]>([]);
  const [retentionDays, setRetentionDays] = useState<number>(3);

  // Thêm state cho Modal CRUD
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingDoc, setEditingDoc] = useState<any>(null);
  const [formData, setFormData] = useState({ id: "", name: "", link: "", content: "{}" });
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [nlText, setNlText] = useState("");
  const [isConverting, setIsConverting] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(() => document.documentElement.classList.contains("dark"));
  const [adminEmails, setAdminEmails] = useState<string[]>([]);
  const [newAdminEmail, setNewAdminEmail] = useState("");

  // State cho thanh tìm kiếm
  const [userSearchTerm, setUserSearchTerm] = useState("");
  const [docSearchTerm, setDocSearchTerm] = useState("");
  const [adminSearchTerm, setAdminSearchTerm] = useState("");

  // Hàm fetch dữ liệu chung có kèm theo Token của Firebase
  const fetchAdminData = async (endpoint: string, options: RequestInit = {}) => {
    const user = auth.currentUser;
    if (!user) return null;
    const token = await user.getIdToken();
    const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
    
    const res = await fetch(`${apiUrl}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
        "Authorization": `Bearer ${token}`,
        "ngrok-skip-browser-warning": "true"
      }
    });
    
    if (!res.ok) {
      if (res.status === 403) alert("Bạn không có quyền truy cập trang quản trị!");
      throw new Error(`HTTP Error: ${res.status}`);
    }
    return res.json();
  };

  // Tải thống kê
  const loadStats = async () => {
    setLoading(true);
    try {
      const data = await fetchAdminData("/admin/stats");
      if (data) setStats(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  // Tải danh sách Người dùng
  const loadUsers = async () => {
    setLoading(true);
    try {
      const data = await fetchAdminData("/admin/users");
      if (data) setUsers(data.users || []);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  // Tải danh sách tài liệu
  const loadDocuments = async () => {
    setLoading(true);
    try {
      const data = await fetchAdminData("/admin/documents");
      if (data) setDocuments(data.documents || []);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  // Tải danh sách backup
  const loadBackups = async () => {
    try {
      const data = await fetchAdminData("/admin/backups");
      if (data) setBackups(data.backups || []);
      const settings = await fetchAdminData("/admin/backups/settings");
      if (settings) setRetentionDays(settings.retention_days || 3);
    } catch (e) {
      console.error(e);
    }
  };

  // Tải danh sách admin
  const loadAdmins = async () => {
    setLoading(true);
    try {
      const data = await fetchAdminData("/admin/admins");
      if (data) setAdminEmails(data.admins || []);
    } catch (e) {}
    setLoading(false);
  };

  // Tải phiên chat của một User cụ thể
  const viewUserSessions = async (uid: string) => {
    setSelectedUid(uid);
    setSelectedSessionId(null);
    setSessionMessages([]);
    setLoading(true);
    try {
      const data = await fetchAdminData(`/admin/users/${uid}/sessions`);
      if (data) setSessions(data.sessions || []);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  // Tải chi tiết một session
  const viewSessionDetail = async (sessionId: string) => {
    setSelectedSessionId(sessionId);
    setSessionMessages([]);
    try {
      const data = await fetchAdminData(`/admin/history/sessions/${sessionId}`);
      if (data) setSessionMessages(data.messages);
    } catch (e) {
      console.error(e);
    }
  };

  // Xóa một phiên chat
  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (!confirm("Bạn có chắc chắn muốn xoá phiên chat này? Hành động này không thể hoàn tác.")) return;
    try {
      await fetchAdminData(`/admin/history/sessions/${sessionId}`, { method: "DELETE" });
      // Cập nhật lại UI sau khi xóa
      setSessions(sessions.filter(s => s.session_id !== sessionId));
      if (selectedSessionId === sessionId) setSessionMessages([]);
    } catch (e) {
      console.error(e);
    }
  };

  // Mở modal thêm/sửa
  const handleOpenModal = (doc: any = null) => {
    setNlText("");
    if (doc) {
      setEditingDoc(doc);
      setFormData({
        id: String(doc.id),
        name: doc.name || "",
        link: doc.link || "",
        content: JSON.stringify(doc.content || {}, null, 2)
      });
    } else {
      setEditingDoc(null);
      setFormData({
        id: `DOC_${Date.now()}`,
        name: "",
        link: "",
        content: "{\n  \"Lĩnh vực\": \"\",\n  \"Trình tự thực hiện\": \"\"\n}"
      });
    }
    setIsModalOpen(true);
  };

  // Hàm chuyển đổi NL sang JSON bằng AI
  const handleConvertNL = async () => {
    if (!nlText.trim()) {
      alert("Vui lòng nhập văn bản tự nhiên để chuyển đổi!");
      return;
    }
    setIsConverting(true);
    try {
      const data = await fetchAdminData("/admin/convert-json", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: nlText })
      });
      if (data && data.status === "success") {
        try {
          const formattedJson = JSON.stringify(JSON.parse(data.data), null, 2);
          setFormData(prev => ({ ...prev, content: formattedJson }));
        } catch (e) {
          setFormData(prev => ({ ...prev, content: data.data }));
        }
        alert("Chuyển đổi thành công! Vui lòng kiểm tra lại nội dung JSON.");
      } else {
        alert(data?.message || "Lỗi khi chuyển đổi");
      }
    } catch (error) {
      console.error(error);
      alert("Đã xảy ra lỗi khi gọi AI chuyển đổi.");
    }
    setIsConverting(false);
  };

  // Lưu tài liệu
  const handleSaveDocument = async () => {
    try {
      const parsedContent = JSON.parse(formData.content);
      const payload = {
        id: formData.id,
        name: formData.name,
        link: formData.link,
        content: parsedContent
      };

      setLoading(true);
      if (editingDoc) {
        await fetchAdminData(`/admin/documents/${editingDoc.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        alert("Cập nhật thành công!");
      } else {
        await fetchAdminData(`/admin/documents`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        alert("Thêm mới thành công!");
      }
      setIsModalOpen(false);
      loadDocuments();
    } catch (error) {
      alert("Lỗi: Kiểm tra lại định dạng JSON hoặc kết nối mạng.");
      console.error(error);
    }
    setLoading(false);
  };

  // Xóa tài liệu
  const handleDeleteDocument = async (docId: string) => {
    if (!confirm("Bạn có chắc muốn xóa thủ tục này khỏi CSDL?")) return;
    setLoading(true);
    try {
      await fetchAdminData(`/admin/documents/${docId}`, { method: "DELETE" });
      alert("Đã xóa thành công!");
      loadDocuments();
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  // Đồng bộ dữ liệu
  const handleSyncDatabase = async () => {
    if (!confirm("Bạn có chắc muốn chạy đồng bộ lại Cơ sở dữ liệu Vector?")) return;
    setLoading(true);
    try {
      const data = await fetchAdminData("/admin/reload-vectordb", { method: "POST" });
      if (data) alert(data.message);
      loadBackups();
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  // Xóa backup
  const handleDeleteBackup = async (backupName: string) => {
    if (!confirm(`Bạn có chắc muốn xóa bản sao lưu ${backupName}?`)) return;
    try {
      await fetchAdminData(`/admin/backups/${backupName}`, { method: "DELETE" });
      alert("Xóa thành công!");
      loadBackups();
    } catch (e) {
      console.error(e);
      alert("Lỗi khi xóa backup!");
    }
  };

  // Lưu cài đặt retention
  const handleSaveRetention = async (days: number) => {
    setRetentionDays(days);
    try {
      await fetchAdminData("/admin/backups/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ retention_days: days })
      });
      alert(`Đã lưu cài đặt tự động xóa sau ${days === -1 ? 'Không bao giờ' : days + ' ngày'}.`);
    } catch (e) {
      console.error(e);
      alert("Lỗi khi lưu cài đặt!");
    }
  };

  // Quản lý Admin
  const handleAddAdmin = async () => {
    if (!newAdminEmail.trim()) return;
    setLoading(true);
    try {
      const res = await fetchAdminData("/admin/admins", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email: newAdminEmail }) });
      if(res) alert(res.message);
      setNewAdminEmail("");
      loadAdmins();
    } catch (e) { alert("Lỗi khi thêm"); }
    setLoading(false);
  };

  const handleRemoveAdmin = async (email: string) => {
    if (!confirm(`Bạn có chắc muốn thu hồi quyền Admin của ${email}?`)) return;
    try {
      const res = await fetchAdminData(`/admin/admins/${email}`, { method: "DELETE" });
      if(res) alert(res.message);
      loadAdmins();
    } catch (e) { alert("Lỗi khi xóa"); }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle("dark");
  };

  useEffect(() => {
    if (activeTab === "stats") loadStats();
    else if (activeTab === "history") loadUsers();
    else if (activeTab === "documents") { loadDocuments(); loadBackups(); }
    else if (activeTab === "admins") loadAdmins();
  }, [activeTab]);

  return (
    <div className="flex h-screen w-full bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-gray-100 font-sans transition-colors duration-500 overflow-hidden relative">
      
      {/* Lớp phủ Mobile */}
      {isSidebarOpen && <div className="fixed inset-0 bg-black/50 z-40 md:hidden" onClick={() => setIsSidebarOpen(false)}></div>}

      {/* Sidebar Admin */}
      <div className={`fixed md:relative z-50 h-full w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col shrink-0 transform transition-transform duration-300 ${isSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}`}>
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h1 className="text-xl font-bold text-blue-600 dark:text-blue-400">Admin Panel</h1>
          <p className="text-xs text-gray-500 mt-1">Hệ thống tra cứu thủ tục</p>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <button
            onClick={() => { setActiveTab("stats"); setIsSidebarOpen(false); }}
            className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${activeTab === "stats" ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 font-medium" : "hover:bg-gray-100 dark:hover:bg-gray-700"}`}
          >
            📊 Thống kê hệ thống
          </button>
          <button
            onClick={() => { setActiveTab("history"); setIsSidebarOpen(false); }}
            className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${activeTab === "history" ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 font-medium" : "hover:bg-gray-100 dark:hover:bg-gray-700"}`}
          >
            🕵️ Giám sát lịch sử
          </button>
          <button
            onClick={() => { setActiveTab("documents"); setIsSidebarOpen(false); }}
            className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${activeTab === "documents" ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 font-medium" : "hover:bg-gray-100 dark:hover:bg-gray-700"}`}
          >
            📚 Quản lý tài liệu & CSDL
          </button>
          <button
            onClick={() => { setActiveTab("admins"); setIsSidebarOpen(false); }}
            className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${activeTab === "admins" ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 font-medium" : "hover:bg-gray-100 dark:hover:bg-gray-700"}`}
          >
            🔐 Phân quyền
          </button>
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative min-w-0">
        
        {/* Header (Chỉ có trong Quản trị để hiển thị Dark Mode & Chuyển Chat) */}
        <div className="h-14 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-4 shrink-0 shadow-sm z-30 relative">
          <button onClick={() => setIsSidebarOpen(true)} className="md:hidden p-2 text-gray-600 dark:text-gray-300">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
          </button>
          <div className="flex-1"></div>
          <div className="flex items-center gap-3">
            <button onClick={toggleDarkMode} className="p-2 rounded-full border border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-all duration-300" title="Chế độ Giao diện">
              {darkMode ? "☀️" : "🌙"}
            </button>
            <button onClick={() => navigate("/chat")} className="bg-green-500 hover:bg-green-600 text-white px-4 py-1.5 rounded-lg text-sm font-medium shadow-sm transition-all duration-300">
              Vào khung Chat
            </button>
            <UserMenu />
          </div>
        </div>

        {/* Nội dung Tab */}
        <div className="flex-1 overflow-auto p-4 md:p-8 relative">
        {loading && (
          <div className="absolute top-4 right-4 bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium animate-pulse shadow">
            Đang tải dữ liệu...
          </div>
        )}

        {/* Tab: Thống kê */}
        {activeTab === "stats" && (
          <div>
            <h2 className="text-2xl font-bold mb-6">Thống kê tổng quan</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-gray-500 dark:text-gray-400 text-sm font-medium">Tổng số Người dùng</h3>
                <p className="text-3xl font-bold mt-2 text-blue-600 dark:text-blue-400">{stats?.total_users || 0}</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-gray-500 dark:text-gray-400 text-sm font-medium">Tổng số Phiên chat</h3>
                <p className="text-3xl font-bold mt-2 text-green-600 dark:text-green-400">{stats?.total_sessions || 0}</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-gray-500 dark:text-gray-400 text-sm font-medium">Trạng thái hệ thống</h3>
                <p className="text-3xl font-bold mt-2 text-purple-600 dark:text-purple-400">{stats?.status || "Unknown"}</p>
              </div>
            </div>

            {/* BIỂU ĐỒ RECHARTS */}
            <div className="mt-8 bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-6">Lưu lượng truy cập (7 ngày qua)</h3>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stats?.chart_data || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                    <YAxis stroke="#6b7280" fontSize={12} allowDecimals={false} />
                    <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                    <Line type="monotone" dataKey="sessions" name="Phiên chat" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Tab: Giám sát lịch sử */}
        {activeTab === "history" && (
          <div className="flex flex-col md:flex-row gap-4 h-[calc(100vh-160px)]">
            
            {/* Cột 1: Danh sách Người dùng */}
            <div className="w-full md:w-1/4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden flex flex-col shadow-sm max-h-60 md:max-h-full">
              <div className="p-4 bg-gray-50 dark:bg-gray-700 font-semibold border-b border-gray-200 dark:border-gray-600 flex justify-between">
                <span>Người dùng</span>
                <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">{users.length}</span>
              </div>
              <div className="p-2 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                <input 
                  type="text" 
                  placeholder="Tìm người dùng..." 
                  value={userSearchTerm}
                  onChange={e => setUserSearchTerm(e.target.value)}
                  className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
              <div className="flex-1 overflow-y-auto">
                {users.filter(u => u.displayName?.toLowerCase().includes(userSearchTerm.toLowerCase()) || u.email?.toLowerCase().includes(userSearchTerm.toLowerCase())).map((u) => (
                  <div 
                    key={u.uid} 
                    onClick={() => viewUserSessions(u.uid)}
                    className={`p-4 border-b border-gray-100 dark:border-gray-700 cursor-pointer transition-colors flex items-center gap-3 ${selectedUid === u.uid ? 'bg-blue-50 dark:bg-blue-900/30' : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'}`}
                  >
                    {u.photoURL ? (
                      <img src={u.photoURL} alt="avatar" className="w-10 h-10 rounded-full object-cover shadow-sm shrink-0" />
                    ) : (
                      <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-sm shrink-0">
                        {u.displayName?.charAt(0).toUpperCase() || u.email?.charAt(0).toUpperCase()}
                      </div>
                    )}
                    <div className="overflow-hidden">
                      <h4 className="font-medium text-sm truncate">{u.displayName}</h4>
                      <p className="text-xs text-gray-500 mt-1 truncate">{u.email}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Cột 2: Danh sách Phiên chat */}
            <div className="w-full md:w-1/4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden flex flex-col shadow-sm max-h-60 md:max-h-full">
              <div className="p-4 bg-gray-50 dark:bg-gray-700 font-semibold border-b border-gray-200 dark:border-gray-600">
                Phiên chat
              </div>
              <div className="flex-1 overflow-y-auto">
                {!selectedUid && <p className="text-gray-500 text-center mt-10 text-sm">Vui lòng chọn một người dùng bên trái</p>}
                {selectedUid && sessions.length === 0 && <p className="text-gray-500 text-center mt-10 text-sm">Chưa có lịch sử</p>}
                {sessions.map((session) => (
                  <div 
                    key={session.session_id} 
                    onClick={() => viewSessionDetail(session.session_id)}
                    className={`p-4 border-b border-gray-100 dark:border-gray-700 cursor-pointer transition-colors flex justify-between items-center group ${selectedSessionId === session.session_id ? 'bg-blue-50 dark:bg-blue-900/30' : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'}`}
                  >
                    <div className="overflow-hidden">
                      <h4 className="font-medium text-sm truncate pr-2">{session.title}</h4>
                    </div>
                    <button 
                      onClick={(e) => handleDeleteSession(e, session.session_id)}
                      className="text-red-500 hover:text-red-700 opacity-0 group-hover:opacity-100 transition-opacity p-2 rounded-full hover:bg-red-50 dark:hover:bg-red-900/20"
                      title="Xóa phiên chat"
                    >
                      🗑️
                    </button>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Cột 3: Nội dung trò chuyện */}
            <div className="flex-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl flex flex-col overflow-hidden shadow-sm min-w-full md:min-w-[300px]">
               <div className="p-4 bg-gray-50 dark:bg-gray-700 font-semibold border-b border-gray-200 dark:border-gray-600">
                Nội dung trò chuyện
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {!selectedSessionId && <p className="text-gray-500 text-center mt-10">Chọn một phiên chat để xem chi tiết</p>}
                {sessionMessages.map((msg) => (
                  <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[75%] rounded-lg p-3 text-sm ${msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}>
                      {msg.content}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tab: Quản lý tài liệu & CSDL */}
        {activeTab === "documents" && (
          <div>
            <h2 className="text-2xl font-bold mb-6">Quản lý Tài liệu & Cơ sở dữ liệu</h2>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 mb-8">
              <h3 className="text-lg font-semibold mb-2">Đồng bộ Vector Database</h3>
              <p className="text-gray-500 dark:text-gray-400 mb-6 text-sm">
                Tính năng này sẽ tiến hành đọc lại dữ liệu từ <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">data/procedures.json</code>, phân mảnh (chunking) và cập nhật embeddings mới vào Vector Database. Quá trình này có thể tốn vài phút.
              </p>
            <div className="flex flex-wrap gap-4">
                <button 
                  onClick={handleSyncDatabase}
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors shadow-sm disabled:opacity-50"
                >
                  Bắt đầu đồng bộ
                </button>
                <button 
                  onClick={() => handleOpenModal()}
                  disabled={loading}
                  className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-lg transition-colors shadow-sm disabled:opacity-50 flex items-center gap-2"
                >
                  <span>➕</span> Thêm tài liệu mới
                </button>
              </div>
            </div>

            {/* QUẢN LÝ BACKUP */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 mb-8">
              <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3 mb-4">
                <h3 className="text-lg font-semibold">Quản lý bản sao lưu (Backup)</h3>
                <div className="flex flex-wrap items-center gap-2">
                  <label className="text-sm text-gray-600 dark:text-gray-400">Tự động xóa sau:</label>
                  <select 
                    value={retentionDays}
                    onChange={(e) => handleSaveRetention(Number(e.target.value))}
                    className="border border-gray-300 dark:border-gray-600 rounded p-1 bg-gray-50 dark:bg-gray-700 text-sm"
                  >
                    <option value={3}>3 ngày</option>
                    <option value={7}>7 ngày</option>
                    <option value={15}>15 ngày</option>
                    <option value={28}>28 ngày</option>
                    <option value={-1}>Không bao giờ xóa</option>
                  </select>
                </div>
              </div>
              
              <div className="overflow-x-auto max-h-[300px]">
                <table className="w-full text-left text-sm whitespace-nowrap border-collapse">
                  <thead className="uppercase text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 sticky top-0">
                    <tr>
                      <th className="px-4 py-2 font-medium">Tên thư mục</th>
                      <th className="px-4 py-2 font-medium text-right">Dung lượng</th>
                      <th className="px-4 py-2 font-medium text-center">Hành động</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {backups.map((bk, idx) => (
                      <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                        <td className="px-4 py-3 font-mono text-xs text-blue-600 dark:text-blue-400">{bk.name}</td>
                        <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">{bk.size}</td>
                        <td className="px-4 py-3 text-center">
                          <button 
                            onClick={() => handleDeleteBackup(bk.name)}
                            className="text-red-500 hover:text-red-700 px-2 py-1 rounded hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
                          >
                            Xóa
                          </button>
                        </td>
                      </tr>
                    ))}
                    {backups.length === 0 && (
                      <tr><td colSpan={3} className="px-4 py-6 text-center text-gray-500">Không có bản sao lưu nào</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* BẢNG DANH SÁCH THỦ TỤC */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="p-4 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600 font-semibold flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3">
                <span>Danh sách Thủ tục pháp luật ({documents.length})</span>
                <input 
                  type="text" 
                  placeholder="Tìm thủ tục (Tên, ID, Lĩnh vực)..." 
                  value={docSearchTerm}
                  onChange={e => setDocSearchTerm(e.target.value)}
                  className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 sm:w-64 font-normal"
                />
              </div>
              <div className="overflow-x-auto max-h-[500px]">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="uppercase text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 sticky top-0 shadow-sm">
                    <tr>
                      <th className="px-6 py-3 font-medium">ID</th>
                      <th className="px-6 py-3 font-medium">Tên thủ tục</th>
                      <th className="px-6 py-3 font-medium">Lĩnh vực</th>
                      <th className="px-6 py-3 font-medium text-center">Nguồn</th>
                      <th className="px-6 py-3 font-medium text-center">Hành động</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {documents.filter(doc => doc.name?.toLowerCase().includes(docSearchTerm.toLowerCase()) || doc.id?.toLowerCase().includes(docSearchTerm.toLowerCase()) || doc.linh_vuc?.toLowerCase().includes(docSearchTerm.toLowerCase())).map((doc, idx) => (
                      <tr 
                        key={idx} 
                        onClick={() => setSelectedDocId(doc.id === selectedDocId ? null : doc.id)}
                        className={`hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors cursor-pointer ${selectedDocId === doc.id ? 'bg-blue-50 dark:bg-blue-900/20' : ''}`}
                      >
                        <td className="px-6 py-4 font-mono text-xs text-gray-500">{doc.id}</td>
                        <td className="px-6 py-4 font-medium text-blue-600 dark:text-blue-400 max-w-[200px] sm:max-w-xs md:max-w-md truncate" title={doc.name}>{doc.name}</td>
                        <td className="px-6 py-4 text-gray-600 dark:text-gray-400 max-w-40 truncate" title={doc.linh_vuc || undefined}>{doc.linh_vuc || "Chung"}</td>
                        <td className="px-6 py-4 text-center">
                          <a href={doc.link} target="_blank" rel="noreferrer" className="text-gray-400 hover:text-blue-500" onClick={(e) => e.stopPropagation()}>🔗 Link</a>
                        </td>
                        <td className="px-6 py-4 text-center whitespace-nowrap min-w-[120px]">
                          {selectedDocId === doc.id && (
                            <div className="flex justify-center items-center">
                              <button onClick={(e) => { e.stopPropagation(); handleOpenModal(doc); }} className="text-blue-500 hover:text-blue-700 mr-3 px-2 py-1 rounded hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors">Sửa</button>
                              <button onClick={(e) => { e.stopPropagation(); handleDeleteDocument(doc.id); }} className="text-red-500 hover:text-red-700 px-2 py-1 rounded hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors">Xóa</button>
                            </div>
                          )}
                        </td>
                      </tr>
                    ))}
                    {documents.filter(doc => doc.name?.toLowerCase().includes(docSearchTerm.toLowerCase()) || doc.id?.toLowerCase().includes(docSearchTerm.toLowerCase()) || doc.linh_vuc?.toLowerCase().includes(docSearchTerm.toLowerCase())).length === 0 && !loading && (
                      <tr><td colSpan={4} className="px-6 py-8 text-center text-gray-500">Không tìm thấy tài liệu nào</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Tab: Phân quyền Quản trị viên */}
        {activeTab === "admins" && (
          <div>
            <h2 className="text-2xl font-bold mb-6">Quản lý Quản trị viên (Admins)</h2>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 max-w-2xl">
              <div className="flex flex-col sm:flex-row gap-3 mb-4">
                <input 
                  type="email" 
                  value={newAdminEmail} 
                  onChange={e => setNewAdminEmail(e.target.value)}
                  placeholder="Nhập email cần cấp quyền..."
                  className="flex-1 border border-gray-300 dark:border-gray-600 rounded-lg p-2 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
                <button onClick={handleAddAdmin} disabled={loading} className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium shadow-sm transition-colors whitespace-nowrap">Thêm Admin</button>
              </div>
              <div className="mb-4">
                <input 
                  type="text" 
                  value={adminSearchTerm} 
                  onChange={e => setAdminSearchTerm(e.target.value)}
                  placeholder="Tìm quản trị viên..."
                  className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                />
              </div>
              <table className="w-full text-left text-sm border-collapse rounded-lg overflow-hidden">
                <thead className="bg-gray-100 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
                  <tr><th className="px-4 py-3 font-medium">Email</th><th className="px-4 py-3 font-medium text-center w-24">Hành động</th></tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  <tr className="bg-gray-50 dark:bg-gray-800"><td className="px-4 py-3 font-medium text-blue-600 dark:text-blue-400 break-all">ngvinh7021@gmail.com <span className="text-[10px] bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400 px-2 py-0.5 rounded-full ml-2 whitespace-nowrap">Super Admin</span></td><td className="px-4 py-3 text-center text-gray-400">-</td></tr>
                  {adminEmails.filter(email => email.toLowerCase().includes(adminSearchTerm.toLowerCase())).map(email => (
                    <tr key={email} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                      <td className="px-4 py-3 text-gray-700 dark:text-gray-300 break-all">{email}</td>
                      <td className="px-4 py-3 text-center">
                        <button onClick={() => handleRemoveAdmin(email)} className="text-red-500 hover:text-red-700 px-2 py-1 rounded hover:bg-red-50 dark:hover:bg-red-900/20">Xóa</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        </div>
      </div>

      {/* Modal Thêm/Sửa Tài liệu */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
              <h3 className="text-lg font-bold">{editingDoc ? "Sửa tài liệu" : "Thêm tài liệu mới"}</h3>
              <button onClick={() => setIsModalOpen(false)} className="text-gray-500 hover:text-red-500">
                ✕
              </button>
            </div>
            <div className="p-4 overflow-y-auto flex-1 space-y-4">
              
              {/* Thêm phần text tự nhiên */}
              <div className="border border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-900/10 p-4 rounded-lg mb-4">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-2">
                  <label className="block text-sm font-bold text-blue-700 dark:text-blue-400">Trợ lý AI: Chuyển văn bản thành JSON</label>
                  <button 
                    onClick={handleConvertNL}
                    disabled={isConverting}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded shadow-sm disabled:opacity-50 flex items-center gap-1"
                  >
                    {isConverting ? "Đang xử lý..." : "✨ Chuyển đổi"}
                  </button>
                </div>
                <textarea 
                  value={nlText} 
                  onChange={e => setNlText(e.target.value)}
                  placeholder="Dán nội dung thủ tục dạng văn bản tự nhiên vào đây để AI tự động trích xuất thành JSON..."
                  rows={4}
                  className="w-full border border-blue-200 dark:border-blue-800 rounded p-2 text-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400" 
                ></textarea>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Mã ID (Duy nhất)</label>
                <input 
                  type="text" 
                  value={formData.id} 
                  onChange={e => setFormData({...formData, id: e.target.value})}
                  disabled={!!editingDoc}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded p-2 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50" 
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Tên thủ tục</label>
                <input 
                  type="text" 
                  value={formData.name} 
                  onChange={e => setFormData({...formData, name: e.target.value})}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded p-2 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100" 
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Đường dẫn tham khảo (Link)</label>
                <input 
                  type="text" 
                  value={formData.link} 
                  onChange={e => setFormData({...formData, link: e.target.value})}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded p-2 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100" 
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Nội dung chi tiết (Định dạng JSON)</label>
                <textarea 
                  value={formData.content} 
                  onChange={e => setFormData({...formData, content: e.target.value})}
                  rows={10}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded p-2 font-mono text-sm bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100" 
                ></textarea>
                <p className="text-xs text-gray-500 mt-1">Lưu ý: Nội dung phải là một JSON hợp lệ chứa các trường như: Lĩnh vực, Thời hạn giải quyết, Trình tự thực hiện,...</p>
              </div>
            </div>
            <div className="p-4 border-t border-gray-200 dark:border-gray-700 flex justify-end gap-3">
              <button onClick={() => setIsModalOpen(false)} className="px-4 py-2 bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium">
                Hủy
              </button>
              <button onClick={handleSaveDocument} disabled={loading} className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors font-medium disabled:opacity-50">
                {loading ? "Đang lưu..." : "Lưu lại"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}