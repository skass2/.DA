import React, { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import UserMenu from '../components/UserMenu';

type ProcedureSearchItem = {
  id: string;
  search_code?: string;
  name: string;
  linh_vuc?: string;
  cap_thuc_hien?: string;
  co_quan?: string;
  ket_qua?: string;
  file_mau_count?: number;
};

function cleanText(value: unknown): string {
  if (value === undefined || value === null) return '';
  return String(value).replace(/\s+/g, ' ').trim();
}

function validResult(item: any): item is ProcedureSearchItem {
  return Boolean(item && cleanText(item.id) && cleanText(item.name));
}

const HomePage = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ProcedureSearchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [showTopBtn, setShowTopBtn] = useState(false);
  const [darkMode, setDarkMode] = useState(() => document.documentElement.classList.contains('dark'));

  const navigate = useNavigate();
  const location = useLocation();

  const fetchProcedures = async (searchQuery: string = '') => {
    setLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const res = await fetch(`${apiUrl}/user/procedures/search?q=${encodeURIComponent(searchQuery)}`, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
      });
      const data = await res.json();
      setResults((data.results || []).filter(validResult));
      setCurrentPage(1);
    } catch (error) {
      console.error('Lỗi tìm kiếm:', error);
      setResults([]);
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

  const totalItems = results.length;
  const totalPages = Math.ceil(totalItems / itemsPerPage);
  const currentItems = useMemo(() => {
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    return results.slice(indexOfFirstItem, indexOfLastItem);
  }, [results, currentPage, itemsPerPage]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    const listElement = document.getElementById('procedure-scroll-container');
    if (listElement) listElement.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setShowTopBtn(e.currentTarget.scrollTop > 300);
  };

  const scrollToTop = () => {
    const listElement = document.getElementById('procedure-scroll-container');
    if (listElement) listElement.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const toggleDarkMode = () => {
    const next = !darkMode;
    setDarkMode(next);
    document.documentElement.classList.toggle('dark', next);
  };

  const getPageNumbers = () => {
    const pages = [];
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, currentPage + 2);

    if (currentPage <= 3) endPage = Math.min(totalPages, 5);
    if (currentPage >= totalPages - 2) startPage = Math.max(1, totalPages - 4);

    for (let i = startPage; i <= endPage; i += 1) pages.push(i);
    return pages;
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4 md:p-8 flex flex-col items-center transition-colors duration-300">
      <div className="w-full max-w-5xl flex flex-col gap-5 h-[90vh]">
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4 shrink-0">
          <h1 className="text-2xl md:text-3xl font-bold text-blue-600 dark:text-blue-400 text-center sm:text-left">
            Cổng Tra Cứu Thủ Tục
          </h1>

          <div className="flex items-center justify-center sm:justify-end gap-3 w-full sm:w-auto">
            <button
              onClick={toggleDarkMode}
              className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 transition-colors text-sm font-semibold"
            >
              {darkMode ? "☀️" : "🌙"}
            </button>

            <Link
              to="/chat"
              className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg shadow-md font-medium transition-colors whitespace-nowrap"
            >
              Trợ lý AI
            </Link>

            <UserMenu />
          </div>
        </div>

        <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3 shrink-0">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Nhập tên, mã, lĩnh vực hoặc cơ quan thực hiện..."
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

        <div className="flex flex-col sm:flex-row justify-between items-center text-sm text-gray-600 dark:text-gray-300 shrink-0 bg-white dark:bg-gray-800 p-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 transition-colors">
          <div>
            Tìm thấy <span className="font-bold text-blue-600 dark:text-blue-400">{totalItems}</span> thủ tục hợp lệ
          </div>

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

        <div
          id="procedure-scroll-container"
          className="flex-1 overflow-y-auto pr-2 bg-transparent rounded-lg"
          style={{ scrollbarWidth: 'thin' }}
          onScroll={handleScroll}
        >
          {loading ? (
            <div className="text-center py-10 text-gray-500 dark:text-gray-400 font-medium animate-pulse">
              Đang tải dữ liệu...
            </div>
          ) : currentItems.length > 0 ? (
            <div className="flex flex-col gap-4">
              {currentItems.map((proc) => (
                <div
                  key={proc.id}
                  onClick={() => navigate(`/procedure/${encodeURIComponent(proc.id)}`)}
                  className="bg-white dark:bg-gray-800 p-5 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md hover:border-blue-400 dark:hover:border-blue-500 cursor-pointer transition-all"
                >
                  <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-3">
                    <div>
                      <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100 mb-2 leading-snug">
                        {proc.name}
                      </h3>

                      <div className="flex flex-wrap gap-2">
                        {cleanText(proc.search_code) && (
                          <span className="inline-block bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 text-xs px-3 py-1.5 rounded-full font-medium">
                            Mã: {proc.search_code}
                          </span>
                        )}

                        <span className="inline-block bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs px-3 py-1.5 rounded-full font-medium border border-blue-100 dark:border-blue-800/50">
                          Lĩnh vực: {proc.linh_vuc || 'Chưa phân loại'}
                        </span>

                        {cleanText(proc.cap_thuc_hien) && (
                          <span className="inline-block bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 text-xs px-3 py-1.5 rounded-full font-medium">
                            Cấp: {proc.cap_thuc_hien}
                          </span>
                        )}

                        {(proc.file_mau_count || 0) > 0 && (
                          <span className="inline-block bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-xs px-3 py-1.5 rounded-full font-medium border border-green-100 dark:border-green-800/50">
                            {proc.file_mau_count} file mẫu
                          </span>
                        )}
                      </div>
                    </div>

                    {cleanText(proc.co_quan) && (
                      <p className="text-sm text-gray-500 dark:text-gray-400 md:max-w-xs leading-relaxed">
                        Cơ quan: {proc.co_quan}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-10 text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-700 transition-colors">
              Không tìm thấy thủ tục nào.
            </div>
          )}
        </div>

        {totalPages > 1 && (
          <div className="flex flex-wrap justify-center items-center gap-1.5 sm:gap-2 shrink-0 pt-2">
            <button
              onClick={() => handlePageChange(1)}
              disabled={currentPage === 1}
              className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 font-bold text-gray-600 dark:text-gray-300 transition-colors"
            >
              Đầu
            </button>

            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 text-gray-600 dark:text-gray-300 transition-colors"
            >
              Trước
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
              className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 text-gray-600 dark:text-gray-300 transition-colors"
            >
              Sau
            </button>

            <button
              onClick={() => handlePageChange(totalPages)}
              disabled={currentPage === totalPages}
              className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 font-bold text-gray-600 dark:text-gray-300 transition-colors"
            >
              Cuối
            </button>
          </div>
        )}

        {showTopBtn && (
          <button
            onClick={scrollToTop}
            className="fixed bottom-6 left-1/2 transform -translate-x-1/2 md:left-auto md:translate-x-0 md:right-8 bg-gray-600 hover:bg-gray-700 text-white px-4 py-3 rounded-full shadow-lg transition-all z-50 font-semibold"
          >
            Lên đầu
          </button>
        )}
      </div>
    </div>
  );
};

export default HomePage;
