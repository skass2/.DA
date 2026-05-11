# RAG Chatbot Tra Cứu Thủ Tục Hành Chính

## Demo Web của tôi: https://mychatbot-7021.web.app/ 

---

## 📖 Giới thiệu
Dự án xây dựng Chatbot tra cứu thủ tục hành chính thông minh, ứng dụng mô hình RAG (Retrieval-Augmented Generation) kết hợp với các mô hình ngôn ngữ lớn (LLM) hiện đại như Gemini 2.5 Flash / Pro, Ollama. Chatbot giúp người dùng (người dân, doanh nghiệp) dễ dàng tìm kiếm thông tin về hồ sơ, lệ phí, thời gian và cơ quan giải quyết các thủ tục hành chính tại Việt Nam bằng ngôn ngữ tự nhiên.

---

## ✨ Tính năng nổi bật
- **Xử lý ngôn ngữ tự nhiên (NLP):** Nhận diện ý định người dùng (Intent Classification) để xử lý các câu chào hỏi, hỏi đáp chung hoặc tra cứu chuyên sâu. Tự động viết lại câu hỏi dựa trên lịch sử chat (Context-aware Rewrite).
- **Giao diện Quản trị (Admin Dashboard) toàn diện:**
  - **Thiết kế Responsive:** Tối ưu hóa hiển thị trên mọi thiết bị (máy tính, điện thoại).
  - **Chuyển đổi Dark/Light Mode:** Tùy chỉnh giao diện sáng/tối.
  - **Quản lý Tài liệu & CSDL:** Thêm, sửa, xóa thủ tục hành chính với hỗ trợ AI chuyển đổi văn bản tự nhiên sang JSON.
  - **Quản lý Backup Vector DB:** Sao lưu, khôi phục và thiết lập chính sách tự động xóa bản sao lưu.
  - **Giám sát Lịch sử Chat:** Xem lịch sử chat chi tiết của từng người dùng, có chức năng xóa phiên chat.
  - **Phân quyền Admin:** Thêm, xóa quản trị viên động, lưu trữ trên Firebase.
  - **Tìm kiếm nâng cao:** Tìm kiếm người dùng, thủ tục và quản trị viên dễ dàng.
- **RAG Pipeline Nâng Cao:**
  - Tìm kiếm Vector kết hợp thuật toán MMR (Maximal Marginal Relevance) nhằm tăng độ đa dạng của tài liệu.
  - Lọc theo metadata (Field-based retrieval) để tìm chính xác phần thông tin người dùng cần (Hồ sơ, lệ phí, thời gian...).
  - Chunk dữ liệu theo từng trường để tối ưu hóa context đưa vào LLM.
  - **Reranking:** Xếp hạng lại kết quả tìm kiếm bằng mô hình Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) giúp tăng độ chính xác của ngữ cảnh.
- **Smart LLM Fallback:** Cơ chế tự động chuyển đổi mô hình (Gemini, OpenRouter, Ollama...) khi gặp sự cố mạng hoặc vượt quá hạn mức API (Lỗi 429).
- **Quản lý phiên chat (Session Memory):** Lưu trữ lịch sử hội thoại trên Firebase Firestore theo thời gian thực, phân loại theo người dùng.
- **Xác thực người dùng:** Hệ thống Đăng nhập / Đăng ký an toàn sử dụng Firebase Auth kết hợp gửi mã xác thực OTP qua Email.

---

## 🛠 Công nghệ sử dụng
- **Backend:** Python, FastAPI, LangChain, SentenceTransformers.
- **Frontend:** React, TypeScript, Vite.
- **Cơ sở dữ liệu:** Firebase Firestore (lưu trữ lịch sử chat, thông tin người dùng, OTP), Vector Database (lưu trữ embeddings thủ tục hành chính).
- **Quản lý Admin:** Firebase Authentication, Firebase Firestore (lưu trữ danh sách admin).
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
5. **AI-powered NL to JSON (Admin):** Chuyển đổi văn bản tự nhiên thành JSON cho dữ liệu thủ tục.
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
### 3. Khởi chạy Ngrok để Public Backend Local ra Internet (Tùy chọn)

Trong quá trình phát triển hệ thống, nếu Frontend đã được deploy lên Cloud (Firebase Hosting, Vercel, Netlify,...) nhưng Backend vẫn chạy trên máy local (`localhost`), Frontend sẽ không thể gọi API trực tiếp từ Internet.  

Để giải quyết vấn đề này, có thể sử dụng Ngrok để tạo một tunnel công khai (public tunnel), cho phép Backend local có thể truy cập được từ bên ngoài Internet.

#### Bước 1: Cài đặt Ngrok

Tải và cài đặt Ngrok tại website chính thức:

[Ngrok Official Website](https://ngrok.com)

Sau khi cài đặt, kiểm tra phiên bản bằng lệnh:

```bash
ngrok version
```

---

#### Bước 2: Khởi chạy Backend

Ví dụ Backend FastAPI đang chạy tại port `8000`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

#### Bước 3: Khởi chạy Ngrok

Mở terminal mới và chạy lệnh:

```bash
ngrok http 8000
```

Sau khi khởi chạy thành công, Ngrok sẽ cung cấp một địa chỉ public dạng:

```bash
Forwarding https://xxxx-xx-xx.ngrok-free.app -> http://localhost:8000
```
Hoặc 

```bash
Forwarding https://xxxx-xx-xx.ngrok-free.dev -> http://localhost:8000
```

Ví dụ:

```bash
https://abcd-1234.ngrok-free.app
```
```bash
https://abcd-1234-xyzz.ngrok-free.dev
```
Đây chính là URL để Frontend trên Cloud có thể gửi request tới Backend local.

---

#### Bước 4: Cập nhật API URL trong Frontend

Thay thế địa chỉ API cũ:

```js
http://localhost:8000
```

thành:

```js
https://abcd-1234.ngrok-free.app
```

Ví dụ:

```js
const API_URL = "https://abcd-1234.ngrok-free.app";
```

---

#### Lưu ý

- URL của Ngrok sẽ thay đổi sau mỗi lần khởi động lại đối với tài khoản miễn phí.
- Backend local phải luôn đang hoạt động để tunnel duy trì kết nối.
- Ngrok chỉ nên sử dụng cho mục đích phát triển và kiểm thử, không phù hợp cho môi trường production thực tế.

Ngrok đặc biệt hữu ích trong các trường hợp:

- Demo đồ án
- Kiểm thử Frontend-Backend
- Test API trên thiết bị khác
- Kết nối tạm thời giữa Cloud và máy local

---

#### Kiểm tra hoạt động

Sau khi cấu hình hoàn tất, truy cập:

```bash
https://abcd-1234.ngrok-free.app/docs
```

Nếu giao diện Swagger UI của FastAPI hiển thị thành công, Backend đã được public ra Internet.
