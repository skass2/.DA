import React, { useState, useEffect } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import UserMenu from '../components/UserMenu';

const HomePage = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  
  // State phân trang
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [showTopBtn, setShowTopBtn] = useState(false);
  const [darkMode, setDarkMode] = useState(() => document.documentElement.classList.contains('dark'));
  
  const navigate = useNavigate();
  const location = useLocation();

  const fetchProcedures = async (searchQuery: string = "") => {
    setLoading(true);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiUrl}/user/procedures/search?q=${encodeURIComponent(searchQuery)}`, {
        headers: { "ngrok-skip-browser-warning": "true" }
      });
      const data = await res.json();
      setResults(data.results || []);
      setCurrentPage(1);
    } catch (error) {
      console.error("Lỗi tìm kiếm:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const searchParam = params.get('search');
    if (searchParam !== null) {
      setQuery(searchParam);
      fetchProcedures(searchParam);
    } else {
      fetchProcedures();
    }
  }, [location.search]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchProcedures(query);
  };

  // Tính toán dữ liệu phân trang
  const totalItems = results.length;
  const totalPages = Math.ceil(totalItems / itemsPerPage);
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = results.slice(indexOfFirstItem, indexOfLastItem);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    // Tự động cuộn lên đầu danh sách khi chuyển trang
    const listElement = document.getElementById('procedure-scroll-container');
    if (listElement) {
        listElement.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    if (e.currentTarget.scrollTop > 300) setShowTopBtn(true);
    else setShowTopBtn(false);
  };

  const scrollToTop = () => {
    const listElement = document.getElementById('procedure-scroll-container');
    if (listElement) listElement.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark');
  };

  // Tạo mảng số trang hiển thị dạng [1, 2, 3, 4, 5]
  const getPageNumbers = () => {
    const pages = [];
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, currentPage + 2);
    
    if (currentPage <= 3) {
      endPage = Math.min(totalPages, 5);
    }
    if (currentPage >= totalPages - 2) {
      startPage = Math.max(1, totalPages - 4);
    }
    for (let i = startPage; i <= endPage; i++) {
      pages.push(i);
    }
    return pages;
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4 md:p-8 flex flex-col items-center transition-colors duration-500">
      <div className="w-full max-w-5xl flex flex-col gap-5 h-[90vh]">
        
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4 shrink-0">
          <h1 className="text-2xl md:text-3xl font-bold text-blue-600 dark:text-blue-400 text-center sm:text-left">Cổng Tra Cứu Thủ Tục</h1>
          <div className="flex items-center justify-center sm:justify-end gap-3 w-full sm:w-auto">
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-full border border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-all duration-300 flex items-center justify-center w-10 h-10"
              title={darkMode ? "Bật chế độ sáng" : "Bật chế độ tối"}
            >
              {darkMode ? "☀️" : "🌙"}
            </button>
            <Link 
              to="/chat" 
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg shadow-md font-medium transition-all whitespace-nowrap"
            >
              Trợ lý AI
            </Link>
            <UserMenu />
          </div>
        </div>
        
        {/* Khung tìm kiếm */}
        <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3 shrink-0">
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Nhập thủ tục cần tìm (VD: đăng ký kết hôn, đất đai...)"
            className="flex-1 p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
          />
          <button 
            type="submit" 
            disabled={loading}
            className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Đang tìm...' : 'Tìm kiếm'}
          </button>
        </form>

        {/* Cấu hình hiển thị */}
        <div className="flex flex-col sm:flex-row justify-between items-center text-sm text-gray-600 dark:text-gray-300 shrink-0 bg-white dark:bg-gray-800 p-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 transition-colors">
          <div>Tìm thấy <span className="font-bold text-blue-600 dark:text-blue-400">{totalItems}</span> thủ tục</div>
          <div className="flex items-center gap-2 mt-2 sm:mt-0">
            <label htmlFor="itemsPerPage" className="font-medium">Số lượng mỗi trang:</label>
            <select 
              id="itemsPerPage" 
              value={itemsPerPage} 
              onChange={(e) => {
                setItemsPerPage(Number(e.target.value));
                setCurrentPage(1);
              }}
              className="border border-gray-300 dark:border-gray-600 rounded p-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50 dark:bg-gray-700 transition-colors"
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        </div>

        {/* Khung chứa danh sách cuộn */}
        <div 
          id="procedure-scroll-container"
          className="flex-1 overflow-y-auto pr-2 bg-transparent rounded-lg"
          style={{ scrollbarWidth: 'thin' }} // Trợ giúp giao diện scrollbar cho Firefox
          onScroll={handleScroll}
        >
          {loading ? (
            <div className="text-center py-10 text-gray-500 dark:text-gray-400 font-medium animate-pulse">Đang tải dữ liệu...</div>
          ) : currentItems.length > 0 ? (
            <div className="flex flex-col gap-4">
              {currentItems.map((proc) => (
                <div 
                  key={proc.id} 
                  onClick={() => navigate(`/procedure/${proc.id}`)}
                  className="bg-white dark:bg-gray-800 p-5 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md hover:border-blue-400 dark:hover:border-blue-500 cursor-pointer transition-all"
                >
                  <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100 mb-2 leading-snug">{proc.name}</h3>
                  <span className="inline-block bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs px-3 py-1.5 rounded-full font-medium border border-blue-100 dark:border-blue-800/50">
                    Lĩnh vực: {proc.linh_vuc || "Chưa phân loại"}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-10 text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-700 transition-colors">
              Không tìm thấy thủ tục nào.
            </div>
          )}
        </div>

        {/* Phân trang */}
        {totalPages > 1 && (
          <div className="flex flex-wrap justify-center items-center gap-1.5 sm:gap-2 shrink-0 pt-2">
            <button 
              onClick={() => handlePageChange(1)} 
              disabled={currentPage === 1}
              className="px-2 sm:px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 font-bold text-gray-600 dark:text-gray-300 transition-colors"
            >
              &laquo;
            </button>
            <button 
              onClick={() => handlePageChange(currentPage - 1)} 
              disabled={currentPage === 1}
              className="px-2 sm:px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 text-gray-600 dark:text-gray-300 transition-colors"
            >
              &lsaquo;
            </button>
            
            {getPageNumbers().map(page => (
              <button 
                key={page}
                onClick={() => handlePageChange(page)}
                className={`px-3 sm:px-4 py-1.5 rounded border font-medium transition-colors ${
                  currentPage === page 
                    ? 'bg-blue-600 text-white border-blue-600 shadow-sm' 
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                {page}
              </button>
            ))}

            <button 
              onClick={() => handlePageChange(currentPage + 1)} 
              disabled={currentPage === totalPages}
              className="px-2 sm:px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 text-gray-600 dark:text-gray-300 transition-colors"
            >
              &rsaquo;
            </button>
            <button 
              onClick={() => handlePageChange(totalPages)} 
              disabled={currentPage === totalPages}
              className="px-2 sm:px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 font-bold text-gray-600 dark:text-gray-300 transition-colors"
            >
              &raquo;
            </button>
          </div>
        )}

        {/* Nút Cuộn lên đầu trang */}
        {showTopBtn && (
          <button
            onClick={scrollToTop}
            className="fixed bottom-6 left-1/2 transform -translate-x-1/2 md:left-auto md:translate-x-0 md:right-8 bg-gray-500 hover:bg-gray-600 dark:bg-gray-600 dark:hover:bg-gray-500 text-white p-3 rounded-full shadow-lg transition-all z-50 flex items-center justify-center w-12 h-12"
            title="Lên đầu trang"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" /></svg>
          </button>
        )}
      </div>
    </div>
  );
};

export default HomePage;
