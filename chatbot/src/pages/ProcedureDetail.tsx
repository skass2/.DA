import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

export default function ProcedureDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [procedure, setProcedure] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDetail = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${apiUrl}/user/procedures/${id}`, {
          headers: { "ngrok-skip-browser-warning": "true" }
        });
        const data = await res.json();
        setProcedure(data.procedure);
      } catch (error) {
        console.error("Lỗi tải chi tiết:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
  }, [id]);

  const handleAskBot = () => {
    // Chuyển sang trang Chat và gửi ngầm tên thủ tục qua state
    navigate('/chat', {
      state: {
        anchorMessage: `Tôi muốn hỏi chi tiết thêm về thủ tục: ${procedure?.name}`
      }
    });
  };

  if (loading) return <div className="text-center p-10 font-medium">Đang tải dữ liệu...</div>;
  if (!procedure) return <div className="text-center p-10 text-red-500">Không tìm thấy thủ tục.</div>;

  const { content } = procedure;

  return (
    <div className="min-h-screen bg-gray-50 p-8 pb-24">
      <div className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-md relative">
        <button onClick={() => navigate('/')} className="text-blue-600 hover:underline mb-4 font-semibold">
          &larr; Về Trang chủ
        </button>
        
        <h1 className="text-2xl font-bold text-gray-900 mb-6">{procedure.name}</h1>
        
        {/* Lặp qua các trường chính để hiển thị tự động dựa trên cấu trúc JSON */}
        {['Lĩnh vực', 'Trình tự thực hiện', 'Cách thức thực hiện', 'Thành phần hồ sơ', 'Thời hạn giải quyết', 'Phí', 'Lệ phí'].map((field) => (
          content[field] ? (
            <div key={field} className="mb-6">
              <h3 className="text-lg font-semibold text-blue-800 bg-blue-50 p-2 rounded">{field}</h3>
              <div className="mt-2 text-gray-700 whitespace-pre-wrap leading-relaxed">
                {content[field]}
              </div>
            </div>
          ) : null
        ))}
      </div>

      {/* Nút thả nổi (Floating Button) gọi Chatbot */}
      <button
        onClick={handleAskBot}
        className="fixed bottom-10 right-10 bg-blue-600 text-white px-6 py-4 rounded-full shadow-2xl hover:bg-blue-700 hover:scale-105 transition-transform flex items-center gap-2 font-bold text-lg z-50 border-4 border-white cursor-pointer"
      >
        <span className="text-2xl">💬</span> Hỏi AI về thủ tục này
      </button>
    </div>
  );
}