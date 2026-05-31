import { useState, useRef, useEffect } from 'react';
import { auth, signOut } from '../firebase';
import { onAuthStateChanged, updateProfile, type User as FirebaseUser } from 'firebase/auth';
import { useNavigate, Link } from 'react-router-dom';
import { createPortal } from 'react-dom';

export default function UserMenu() {
  const [firebaseUser, setFirebaseUser] = useState<FirebaseUser | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [newName, setNewName] = useState("");
  const [isUpdating, setIsUpdating] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setFirebaseUser(currentUser);
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (showProfile && firebaseUser) {
      setNewName(firebaseUser.displayName || "");
      setIsEditing(false);
    }
  }, [showProfile, firebaseUser]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    await signOut(auth);
    setShowLogoutConfirm(false);
    navigate('/');
  };

  const handleUpdateName = async () => {
    if (!firebaseUser) return;
    if (!newName.trim()) {
      alert("Tên hiển thị không được để trống!");
      return;
    }
    setIsUpdating(true);
    try {
      await updateProfile(firebaseUser, { displayName: newName.trim() });
      // Cập nhật state cục bộ để UI thay đổi ngay lập tức
      setFirebaseUser({ ...firebaseUser, displayName: newName.trim() } as FirebaseUser);
      setIsEditing(false);
      // Bắn sự kiện để các Component khác (như ChatBox) biết và cập nhật lại
      window.dispatchEvent(new Event('profileUpdated'));
    } catch (error) {
      console.error(error);
      alert("Lỗi khi cập nhật tên!");
    }
    setIsUpdating(false);
  };

  if (!firebaseUser) {
    return (
      <Link 
        to="/login"
        className="px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-800 dark:text-white text-sm font-medium rounded-lg transition-colors shadow-sm whitespace-nowrap"
      >
        Đăng nhập
      </Link>
    );
  }

  return (
    <div className="relative z-50 shrink-0" ref={menuRef}>
      <div 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full border border-gray-200 dark:border-gray-600 shadow-sm transition-colors duration-500 max-w-[150px] sm:max-w-[200px] cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-600"
      >
        {firebaseUser.photoURL ? (
          <img src={firebaseUser.photoURL} alt="Avatar" className="w-6 h-6 sm:w-7 sm:h-7 rounded-full shadow-sm shrink-0 object-cover" />
        ) : (
          <div className="w-6 h-6 sm:w-7 sm:h-7 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-xs sm:text-sm shrink-0">
            {firebaseUser.displayName?.charAt(0).toUpperCase() || firebaseUser.email?.charAt(0).toUpperCase() || 'U'}
          </div>
        )}
        <span className="text-gray-700 dark:text-gray-200 font-medium text-xs sm:text-sm max-w-[80px] sm:max-w-[120px] truncate transition-colors duration-500">
          {firebaseUser.displayName || firebaseUser.email}
        </span>
      </div>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 sm:w-64 bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 py-1 overflow-hidden origin-top-right">
          <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50">
            <p className="text-sm text-gray-900 dark:text-white font-bold truncate">{firebaseUser.displayName || "Người dùng"}</p>
            <p className="text-xs text-gray-500 dark:text-gray-400 truncate mt-0.5">{firebaseUser.email}</p>
          </div>
          <button 
            onClick={() => { setIsOpen(false); setShowProfile(true); }}
            className="w-full text-left px-4 py-2.5 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
          >
            <span>👤</span> Hồ sơ cá nhân
          </button>
          <button 
            onClick={() => { setIsOpen(false); setShowLogoutConfirm(true); }}
            className="w-full text-left px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors flex items-center gap-2"
          >
            <span>🚪</span> Đăng xuất
          </button>
        </div>
      )}

      {/* Modal Profile */}
      {showProfile && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 cursor-default" style={{ zIndex: 9999 }} onClick={() => setShowProfile(false)}>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-sm p-6 relative border border-gray-200 dark:border-gray-700" onClick={e => e.stopPropagation()}>
            <button onClick={() => setShowProfile(false)} className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">✕</button>
            <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-6 text-center">Hồ sơ cá nhân</h3>
            <div className="flex flex-col items-center gap-4 mb-6">
              {firebaseUser.photoURL ? (
                <img src={firebaseUser.photoURL} alt="Avatar" className="w-20 h-20 rounded-full shadow-md object-cover" />
              ) : (
                <div className="w-20 h-20 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-3xl shadow-md">
                  {firebaseUser.displayName?.charAt(0).toUpperCase() || firebaseUser.email?.charAt(0).toUpperCase() || 'U'}
                </div>
              )}
              <div className="text-center w-full px-4">
                {isEditing ? (
                  <div className="flex flex-col items-center gap-2 mt-2">
                    <input 
                      type="text" 
                      value={newName} 
                      onChange={(e) => setNewName(e.target.value)} 
                      className="border border-gray-300 dark:border-gray-600 rounded px-3 py-1.5 text-sm w-full bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      placeholder="Nhập tên mới..."
                      autoFocus
                    />
                    <div className="flex gap-2 justify-center">
                      <button onClick={() => setIsEditing(false)} className="text-xs px-3 py-1.5 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500 transition-colors font-medium">Hủy</button>
                      <button onClick={handleUpdateName} disabled={isUpdating} className="text-xs px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50 font-medium">
                        {isUpdating ? "Đang lưu..." : "Lưu lại"}
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <p className="text-lg font-bold text-gray-900 dark:text-white truncate flex items-center justify-center gap-2">
                      {firebaseUser.displayName || "Người dùng"}
                      <button onClick={() => setIsEditing(true)} className="text-gray-400 hover:text-blue-500 transition-colors shrink-0" title="Đổi tên">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" /></svg>
                      </button>
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 truncate mt-0.5">{firebaseUser.email}</p>
                  </>
                )}
              </div>
            </div>
            <div className="space-y-3">
              <div className="bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg">
                <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Trạng thái xác thực</p>
                <p className="text-sm font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  {firebaseUser.emailVerified ? <span className="text-green-500">✓ Đã xác thực</span> : <span className="text-yellow-500">⚠ Chưa xác thực</span>}
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg">
                <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Ngày tham gia</p>
                <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
                  {firebaseUser.metadata.creationTime ? new Date(firebaseUser.metadata.creationTime).toLocaleDateString('vi-VN') : 'Không rõ'}
                </p>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Modal Xác nhận Đăng xuất */}
      {showLogoutConfirm && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 cursor-default" style={{ zIndex: 9999 }}>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-sm p-6 text-center border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">Xác nhận đăng xuất</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">Bạn có chắc chắn muốn thoát khỏi phiên làm việc?</p>
            <div className="flex justify-center gap-4">
              <button onClick={() => setShowLogoutConfirm(false)} className="px-6 py-2 bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium">
                Hủy
              </button>
              <button onClick={handleLogout} className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium shadow-sm">
                Đăng xuất
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}