# RAG Chatbot Tra Cứu Thủ Tục Hành Chính

## 📖 Giới thiệu
Dự án xây dựng Chatbot tra cứu thủ tục hành chính thông minh, ứng dụng mô hình RAG (Retrieval-Augmented Generation) kết hợp với các mô hình ngôn ngữ lớn (LLM) hiện đại như Gemini 2.5 Flash / Pro, Ollama. Chatbot giúp người dùng (người dân, doanh nghiệp) dễ dàng tìm kiếm thông tin về hồ sơ, lệ phí, thời gian và cơ quan giải quyết các thủ tục hành chính tại Việt Nam bằng ngôn ngữ tự nhiên.

---

## ✨ Tính năng nổi bật
- **Xử lý ngôn ngữ tự nhiên (NLP):** Nhận diện ý định người dùng (Intent Classification) để xử lý các câu chào hỏi, hỏi đáp chung hoặc tra cứu chuyên sâu. Tự động viết lại câu hỏi dựa trên lịch sử chat (Context-aware Rewrite).
- **RAG Pipeline Nâng Cao:**
  - Tìm kiếm Vector kết hợp thuật toán MMR (Maximal Marginal Relevance) nhằm tăng độ đa dạng của tài liệu.
  - Lọc theo metadata (Field-based retrieval) để tìm chính xác phần thông tin người dùng cần (Hồ sơ, lệ phí, thời gian...).
  - Chunk dữ liệu theo từng trường để tối ưu hóa context đưa vào LLM.
  - **Reranking:** Xếp hạng lại kết quả tìm kiếm bằng mô hình Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) giúp tăng độ chính xác của ngữ cảnh.
- **Smart LLM Fallback:** Cơ chế tự động chuyển đổi mô hình (Gemini, OpenRouter, Ollama...) khi gặp sự cố mạng hoặc vượt quá hạn mức API (Lỗi 429).
- **Quản lý phiên chat (Session Memory):** Lưu trữ và phân trang lịch sử hội thoại trên Firebase Firestore theo thời gian thực.
- **Xác thực người dùng:** Hệ thống Đăng nhập / Đăng ký an toàn sử dụng Firebase Auth kết hợp gửi mã xác thực OTP qua Email.

---

## 🛠 Công nghệ sử dụng
- **Backend:** Python, FastAPI, LangChain, SentenceTransformers.
- **Frontend:** React, TypeScript, Vite.
- **Cơ sở dữ liệu:** Firebase Firestore (lưu trữ lịch sử chat, thông tin người dùng, OTP), Vector Database (lưu trữ embeddings thủ tục hành chính).
- **AI / LLM:** Google Gemini, Cross-Encoder (Reranker).

---

## 📂 Cấu trúc dự án
```text
.DA/
├── backend/
│   ├── auth.py             # Middleware xác thực bằng Firebase Admin
│   ├── data/               # Chứa dữ liệu JSON (intents, procedures, entities) và script xử lý
│   ├── rag/                # Lõi xử lý RAG (pipeline.py, memory.py, intent.py, normalizer.py)
│   ├── routers/            # API endpoints cho Auth và User chat
│   └── main.py             # Entry point khởi chạy ứng dụng FastAPI
├── chatbot/                # Mã nguồn Frontend (React + TypeScript + Vite)
│   ├── src/                
│   └── package.json        
└── README.md
```

---

## ⚙️ Kiến trúc hệ thống (RAG Pipeline)
1. **User Query:** Người dùng gửi câu hỏi.
2. **Intent Detection:** Xác định ý định (chào hỏi, hỏi thủ tục, không hiểu,...).
3. **Query Rewrite:** Dùng LLM viết lại câu hỏi dựa trên lịch sử để làm rõ ngữ cảnh.
4. **Normalize & Keyword Extraction:** Chuẩn hóa câu hỏi, trích xuất thực thể (Tên thủ tục, Lĩnh vực).
5. **Vector Search (k=15):** Truy xuất tài liệu từ Vector DB, kết hợp lọc theo metadata.
6. **Reranking:** Chấm điểm và sắp xếp lại các chunk tài liệu bằng Cross-Encoder.
7. **Context Assembly:** Tổng hợp top các chunk tài liệu liên quan nhất làm ngữ cảnh.
8. **LLM Generation:** Nạp Context và Query vào LLM (Gemini/Fallback) để tạo câu trả lời với strict prompt.
9. **Response:** Trả kết quả cho người dùng và lưu lịch sử vào Firestore.

---

## 🚀 Hướng dẫn cài đặt và chạy dự án

### 1. Cài đặt Backend
```bash
cd backend
# Tạo môi trường ảo và kích hoạt
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Khởi chạy server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Cài đặt Frontend
```bash
cd chatbot
npm install
npm run dev
```
