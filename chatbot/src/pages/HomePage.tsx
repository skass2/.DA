import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';

export default function HomePage() {
  const [searchParams] = useSearchParams();
  const initialQuery = searchParams.get("search") || "";
  const [query, setQuery] = useState(initialQuery);
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const executeSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    try {
      // Cập nhật URL gốc theo cấu hình của bạn (hoặc dùng Axios)
      const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiUrl}/user/procedures/search?q=${encodeURIComponent(searchQuery)}`, {
        headers: { "ngrok-skip-browser-warning": "true" }
      });
      const data = await res.json();
      setResults(data.results || []);
    } catch (error) {
      console.error("Lỗi tìm kiếm:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    executeSearch(query);
  };

  useEffect(() => {
    if (initialQuery) {
      executeSearch(initialQuery);
    }
  }, [initialQuery]);

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-5xl mx-auto relative">
        <div className="absolute top-0 right-0">
          <Link 
            to="/chat" 
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg shadow-md font-medium transition-all"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
            Trợ lý AI
          </Link>
        </div>
        <h1 className="text-3xl font-bold text-center text-blue-600 mb-8">Cổng Tra Cứu Thủ Tục Hành Chính</h1>
        
        <form onSubmit={handleSearch} className="mb-8 flex gap-2">
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Nhập thủ tục cần tìm (VD: đăng ký kết hôn, làm căn cước...)"
            className="flex-1 p-4 rounded-lg border border-gray-300 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button 
            type="submit" 
            disabled={loading}
            className="bg-blue-600 text-white px-8 py-4 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Đang tìm...' : 'Tìm kiếm'}
          </button>
        </form>

        <div className="grid gap-4">
          {results.map((proc) => (
            <div 
              key={proc.id} 
              onClick={() => navigate(`/procedure/${proc.id}`)}
              className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 hover:shadow-md cursor-pointer transition-shadow"
            >
              <h3 className="text-xl font-semibold text-gray-800 mb-2">{proc.name}</h3>
              <span className="inline-block bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full">
                Lĩnh vực: {proc.linh_vuc}
              </span>
            </div>
          ))}
          {results.length === 0 && query && !loading && (
            <p className="text-center text-gray-500">Không tìm thấy thủ tục nào phù hợp.</p>
          )}
        </div>
      </div>
    </div>
  );
}