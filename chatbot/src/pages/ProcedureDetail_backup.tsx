import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import UserMenu from '../components/UserMenu';

const ProcedureDetail = () => {
  const { id } = useParams();
  const [procedure, setProcedure] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showTopBtn, setShowTopBtn] = useState(false);
  const [darkMode, setDarkMode] = useState(() => document.documentElement.classList.contains('dark'));
  const navigate = useNavigate();

  useEffect(() => {
    const fetchProcedure = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
        const response = await fetch(`${apiUrl}/user/procedures/${id}`, {
          headers: { "ngrok-skip-browser-warning": "true" }
        });
        const data = await response.json();
        setProcedure(data.procedure || data);
      } catch (error) {
        console.error("Lỗi lấy chi tiết thủ tục:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchProcedure();
  }, [id]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    if (e.currentTarget.scrollTop > 300) setShowTopBtn(true);
    else setShowTopBtn(false);
  };

  const scrollToTop = () => {
    const container = document.getElementById('detail-scroll-container');
    if (container) container.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleAskBot = () => {
    navigate('/chat', {
      state: {
        anchorMessage: `Tôi muốn hỏi chi tiết thêm về thủ tục: ${procedure?.name}`
      }
    });
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark');
  };

  if (loading) return <div className="h-screen flex items-center justify-center font-medium text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-900 transition-colors">Đang tải thông tin thủ tục...</div>;
  if (!procedure) return <div className="h-screen flex items-center justify-center font-medium text-red-500 bg-gray-50 dark:bg-gray-900 transition-colors">Không tìm thấy thủ tục này.</div>;

  const content = procedure.content || {};

  // Hàm render các trường chuỗi cơ bản
  const renderStringField = (label: string, value: any) => {
    if (!value || typeof value !== 'string' || value.trim() === '') return null;
    return (
      <div className="mb-6">
        <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30 px-3 py-1.5 rounded inline-block mb-3 border border-blue-100 dark:border-blue-800/50">{label}</h3>
        <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">{value}</p>
      </div>
    );
  };

  return (
    <div id="detail-scroll-container" onScroll={handleScroll} className="h-screen overflow-y-auto bg-gray-50 dark:bg-gray-900 py-8 relative transition-colors duration-500">
      <div className="max-w-5xl mx-auto p-6 md:p-10 bg-white dark:bg-gray-800 shadow-lg rounded-xl border border-gray-100 dark:border-gray-700 mb-24 transition-colors duration-500">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
          <Link to="/" className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium flex items-center gap-1 transition-colors">
            <span>👈</span> Quay lại Cổng tra cứu
          </Link>
          <div className="flex items-center justify-end gap-3 w-full sm:w-auto">
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-full border border-gray-300 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-all duration-300 flex items-center justify-center w-10 h-10"
              title={darkMode ? "Bật chế độ sáng" : "Bật chế độ tối"}
            >
              {darkMode ? "☀️" : "🌙"}
            </button>
            <UserMenu />
          </div>
        </div>

        <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-gray-100 mb-8 leading-tight">{procedure.name}</h1>
        
        {renderStringField("Lĩnh vực", content["Lĩnh vực"])}
        {renderStringField("Đối tượng thực hiện", content["Đối tượng thực hiện"])}
        {renderStringField("Cơ quan thực hiện", content["Cơ quan thực hiện"])}
        {renderStringField("Cơ quan ban hành", content["Cơ quan ban hành"])}
        {renderStringField("Cơ quan phối hợp", content["Cơ quan phối hợp"])}
        {renderStringField("Kết quả thực hiện", content["Kết quả thực hiện"])}
        {renderStringField("Trình tự thực hiện", content["Trình tự thực hiện"])}
        {renderStringField("Yêu cầu điều kiện", content["Yêu cầu điều kiện"])}

        {/* BẢNG: CÁCH THỨC THỰC HIỆN */}
        {content["Cách thức thực hiện"] && content["Cách thức thực hiện"].length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30 px-3 py-1.5 rounded inline-block mb-4 border border-blue-100 dark:border-blue-800/50">Cách thức thực hiện</h3>
            <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
              <table className="min-w-full text-sm text-left text-gray-600 dark:text-gray-300">
                <thead className="bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 uppercase font-semibold">
                  <tr>
                    <th className="py-3 px-4 border-b w-1/4">Hình thức</th>
                    <th className="py-3 px-4 border-b w-1/4">Thời hạn</th>
                    <th className="py-3 px-4 border-b w-2/4">Mô tả</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {content["Cách thức thực hiện"].map((item: any, idx: number) => (
                    <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                      <td className="py-3 px-4 font-medium text-gray-900 dark:text-gray-100 align-top">{item["Hình thức"]}</td>
                      <td className="py-3 px-4 align-top">{item["Thời hạn"]}</td>
                      <td className="py-3 px-4 align-top whitespace-pre-wrap">{item["Mô tả"]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* DANH SÁCH: THÀNH PHẦN HỒ SƠ */}
        {content["Thành phần hồ sơ"] && content["Thành phần hồ sơ"].length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30 px-3 py-1.5 rounded inline-block mb-4 border border-blue-100 dark:border-blue-800/50">Thành phần hồ sơ</h3>
            <div className="flex flex-col gap-4">
              {content["Thành phần hồ sơ"].map((item: any, idx: number) => (
                <div key={idx} className="bg-gray-50 dark:bg-gray-700/30 p-4 rounded-lg border border-gray-200 dark:border-gray-700 transition-colors">
                  <p className="font-semibold text-gray-800 dark:text-gray-200 mb-2">{idx + 1}. {item["Tên giấy tờ"]}</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm dark:text-gray-400">
                    {item["Số lượng"] && (
                      <div className="flex gap-2">
                        <span className="text-gray-500 dark:text-gray-500">Số lượng:</span>
                        <span className="font-medium whitespace-pre-wrap">{item["Số lượng"]}</span>
                      </div>
                    )}
                    {item["Biểu mẫu"] && (
                      <div className="flex gap-2">
                        <span className="text-gray-500 dark:text-gray-500">Biểu mẫu đính kèm:</span>
                        <span className="font-medium text-blue-600 dark:text-blue-400">{item["Biểu mẫu"]}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* DANH SÁCH: PHÍ / LỆ PHÍ */}
        {(content["Phí"]?.length > 0 || content["Lệ phí"]?.length > 0) && (
          <div className="mb-8">
            <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30 px-3 py-1.5 rounded inline-block mb-4 border border-blue-100 dark:border-blue-800/50">Phí / Lệ phí</h3>
            <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-700/30 p-4 rounded-lg border border-gray-200 dark:border-gray-700 transition-colors">
              {content["Phí"]?.map((item: any, idx: number) => (
                <li key={`fee-${idx}`} className="whitespace-pre-wrap">{item.text}</li>
              ))}
              {content["Lệ phí"]?.map((item: any, idx: number) => (
                <li key={`fee2-${idx}`} className="whitespace-pre-wrap">{item.text}</li>
              ))}
            </ul>
          </div>
        )}

        {/* BẢNG: CĂN CỨ PHÁP LÝ */}
        {content["Căn cứ pháp lý"] && content["Căn cứ pháp lý"].length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30 px-3 py-1.5 rounded inline-block mb-4 border border-blue-100 dark:border-blue-800/50">Căn cứ pháp lý</h3>
            <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
              <table className="min-w-full text-sm text-left text-gray-600 dark:text-gray-300">
                <thead className="bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 uppercase font-semibold">
                  <tr>
                    <th className="py-3 px-4 border-b">Số hiệu</th>
                    <th className="py-3 px-4 border-b">Tên văn bản</th>
                    <th className="py-3 px-4 border-b">Cơ quan ban hành</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {content["Căn cứ pháp lý"].map((item: any, idx: number) => (
                    <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                      <td className="py-3 px-4 font-medium text-blue-700 dark:text-blue-400 align-top whitespace-nowrap">{item["Số hiệu"]}</td>
                      <td className="py-3 px-4 align-top">{item["Tên văn bản"]}</td>
                      <td className="py-3 px-4 align-top">{item["Cơ quan ban hành"]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Nhóm nút thả nổi góc dưới bên phải */}
      <div className="fixed bottom-8 right-8 md:bottom-10 md:right-10 flex flex-col gap-3 z-50">
        {/* Nút Cuộn lên đầu trang */}
        {showTopBtn && (
          <button
            onClick={scrollToTop}
            className="bg-gray-500 hover:bg-gray-600 dark:bg-gray-600 dark:hover:bg-gray-500 text-white p-3 rounded-full shadow-lg transition-all flex items-center justify-center w-12 h-12 self-end"
            title="Lên đầu trang"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" /></svg>
          </button>
        )}
        
        {/* Nút thả nổi (Floating Button) gọi Chatbot */}
        <button
          onClick={handleAskBot}
          className="bg-blue-600 text-white px-5 py-3 md:px-6 md:py-4 rounded-full shadow-2xl hover:bg-blue-700 hover:scale-105 transition-transform flex items-center gap-2 font-bold text-base md:text-lg border-2 md:border-4 border-white dark:border-gray-800 cursor-pointer"
        >
          <span className="text-xl md:text-2xl">💬</span> <span className="hidden sm:inline">Hỏi AI về thủ tục này</span>
        </button>
      </div>
    </div>
  );
};

export default ProcedureDetail;
