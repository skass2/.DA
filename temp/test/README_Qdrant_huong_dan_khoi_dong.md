# RAG Chatbot Tra Cứu Thủ Tục Hành Chính

## Demo Web
https://mychatbot-7021.web.app/

## 1. Giới thiệu

Dự án xây dựng chatbot hỗ trợ tra cứu thủ tục hành chính bằng mô hình RAG (Retrieval-Augmented Generation). Hệ thống cho phép người dùng hỏi bằng ngôn ngữ tự nhiên để tra cứu hồ sơ, lệ phí, thời hạn, cơ quan giải quyết, căn cứ pháp lý và các thông tin liên quan đến thủ tục hành chính.

Phiên bản hiện tại đã chuyển tầng lưu trữ vector sang Qdrant chạy local bằng Docker. Qdrant được dùng để lưu embedding của các chunk thủ tục hành chính, giúp truy xuất ngữ cảnh nhanh và hỗ trợ lọc metadata chính xác theo tên thủ tục, lĩnh vực và loại thông tin.

## 2. Tính năng chính

- Tra cứu thủ tục hành chính theo câu hỏi tự nhiên.
- Tìm kiếm thủ tục theo tên hoặc lĩnh vực.
- Hỏi đáp theo từng nhóm thông tin: hồ sơ, thời hạn, lệ phí, cơ quan thực hiện, kết quả, căn cứ pháp lý.
- RAG pipeline có nhận diện intent, chuẩn hóa câu hỏi, nhận diện tên thủ tục, lọc field và rerank kết quả.
- Qdrant local dùng làm vector database.
- Exact field boost: khi hệ thống đã nhận diện đúng thủ tục và loại thông tin cần hỏi, backend lấy trực tiếp chunk theo metadata để tránh bỏ sót dữ liệu quan trọng.
- Reranking bằng CrossEncoder để tăng độ chính xác của context.
- Gemini 2.5 Flash dùng làm LLM chính, Gemini 2.5 Flash Lite dùng cho rewrite/fallback nhẹ.
- Hỗ trợ Firebase Auth để bảo vệ API chat thật.
- Lưu lịch sử chat theo phiên trên Firebase Firestore.
- Có endpoint dev local `/dev/chat` để test nhanh RAG không cần đăng nhập, chỉ bật khi `ENABLE_DEV_ROUTES=true`.

## 3. Công nghệ sử dụng

### Backend

- Python 3.10
- FastAPI
- Uvicorn
- LangChain
- SentenceTransformers
- Qdrant Client
- LangChain Qdrant
- Firebase Admin SDK
- Python Dotenv

### AI và RAG

- Google Gemini 2.5 Flash
- Google Gemini 2.5 Flash Lite
- CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Embedding model `sentence-transformers/all-MiniLM-L6-v2`

### Vector database

- Qdrant local chạy bằng Docker
- Collection mặc định: `thu_tuc_hanh_chinh_lai_chau`
- Vector size: 384 với model `all-MiniLM-L6-v2`

### Frontend

- React
- TypeScript
- Vite
- Firebase Auth

## 4. Cấu trúc dự án

```text
.DA/
├── backend/
│   ├── data/
│   │   ├── procedures.json
│   │   ├── entities.json
│   │   ├── intents.json
│   │   └── synonyms.json
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── config.py
│   │   ├── intent.py
│   │   ├── loader.py
│   │   ├── memory.py
│   │   ├── normalizer.py
│   │   ├── pipeline.py
│   │   └── vectorstore.py
│   ├── routers/
│   │   ├── admin.py
│   │   ├── auth.py
│   │   ├── dev.py
│   │   └── user.py
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   └── serviceAccountKey.json
├── chatbot/
│   ├── src/
│   └── package.json
└── README.md
```

## 5. Biến môi trường backend

Tạo file `.env` trong thư mục `backend`.

```env
GOOGLE_API_KEY=your_google_api_key

CHATBOT_PRIMARY_MODEL=gemini-2.5-flash
CHATBOT_LIGHTWEIGHT_MODEL=gemini-2.5-flash-lite
CHATBOT_TEMPERATURE=0.2
CHATBOT_TIMEOUT=30

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=thu_tuc_hanh_chinh_lai_chau
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

ENABLE_DEV_ROUTES=false
```

Ghi chú:

- Khi test local nhanh bằng `/dev/chat`, có thể đặt `ENABLE_DEV_ROUTES=true`.
- Khi deploy hoặc demo public, nên đặt `ENABLE_DEV_ROUTES=false` để tắt route test không cần đăng nhập.
- Không commit `.env` và `serviceAccountKey.json` lên GitHub.

## 6. Cài đặt backend

Khuyến nghị dùng Python 3.10 vì các thư viện AI/RAG ổn định hơn với phiên bản này.

```powershell
cd D:\.DA\backend
py -3.10 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra các thư viện chính:

```powershell
python -c "from qdrant_client import QdrantClient; from langchain_qdrant import QdrantVectorStore; from langchain_huggingface import HuggingFaceEmbeddings; from rapidfuzz import fuzz; print('Libraries OK')"
```

## 7. Cài đặt và chạy Qdrant local bằng Docker

Qdrant là vector database dùng để lưu embedding của dữ liệu thủ tục hành chính. Backend hiện tại kết nối tới Qdrant qua địa chỉ:

```env
QDRANT_URL=http://localhost:6333
```

Vì vậy, trước khi chạy backend FastAPI, bố cần đảm bảo Docker Desktop đã mở và container Qdrant đang chạy.

### 7.1. Kiểm tra Docker

Mở Docker Desktop và đợi Docker Engine chạy xong. Sau đó kiểm tra bằng PowerShell:

```powershell
docker ps
```

Nếu lệnh không báo lỗi thì Docker đã sẵn sàng.

### 7.2. Chạy Qdrant lần đầu bằng Docker

Nếu máy chưa có container `qdrant_local`, chạy lệnh sau:

```powershell
docker run -d --name qdrant_local -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Ý nghĩa:

```text
--name qdrant_local
=> đặt tên container là qdrant_local

-p 6333:6333
=> mở cổng REST API và Dashboard của Qdrant

-p 6334:6334
=> mở cổng gRPC của Qdrant
```

Kiểm tra Qdrant API:

```powershell
Invoke-RestMethod http://localhost:6333
Invoke-RestMethod http://localhost:6333/collections
```

Mở dashboard:

```text
http://localhost:6333/dashboard
```

### 7.3. Khởi động lại Qdrant khi container đã tồn tại

Nếu container `qdrant_local` đã được tạo từ trước, mỗi lần mở lại máy bố chỉ cần chạy:

```powershell
docker start qdrant_local
```

Kiểm tra container đang chạy:

```powershell
docker ps
```

Nếu thấy dòng có tên `qdrant_local` và cổng `6333-6334` thì Qdrant đã chạy.

### 7.4. Dùng docker-compose.yml nếu muốn quản lý Qdrant bằng Compose

Nếu thư mục backend có file `docker-compose.yml`, bố có thể dùng Docker Compose để chạy Qdrant:

```powershell
cd D:\.DA\backend
docker compose up -d
```

File `docker-compose.yml` thường có tác dụng:

```text
image: qdrant/qdrant
=> dùng image Qdrant

container_name: qdrant_local
=> đặt tên container là qdrant_local

ports 6333 và 6334
=> mở API, dashboard và gRPC

volume qdrant_storage
=> lưu dữ liệu vector ra ổ đĩa để tắt máy không mất dữ liệu
```

Lưu ý: không nên chạy song song cả `docker run` và `docker compose up` nếu cả hai cùng tạo container tên `qdrant_local`, vì có thể bị trùng tên container hoặc trùng cổng.

Nếu hiện tại container `qdrant_local` đã chạy ổn, có thể tiếp tục dùng:

```powershell
docker start qdrant_local
```

### 7.5. Cho Qdrant tự khởi động cùng Docker

Để Docker tự bật lại container Qdrant sau khi mở máy hoặc sau khi Docker Desktop khởi động, chạy:

```powershell
docker update --restart unless-stopped qdrant_local
```

Nếu muốn Docker Desktop tự mở khi đăng nhập Windows, bật trong Docker Desktop:

```text
Settings -> General -> Start Docker Desktop when you sign in
```

Khi đó quy trình mở máy sẽ nhẹ hơn:

```text
Mở máy
-> Docker Desktop tự chạy
-> qdrant_local tự chạy theo Docker
-> chỉ cần mở backend FastAPI
```


## 8. Nạp dữ liệu thủ tục vào Qdrant

Sau khi Qdrant chạy, build vector database từ `data/procedures.json`:

```powershell
python -c "from rag.loader import load_data; from rag.chunker import create_chunks; from rag.vectorstore import build_vectorstore; data=load_data(); chunks=create_chunks(data); print('START BUILD QDRANT...'); db=build_vectorstore(chunks, backup=True); print('BUILD DONE:', db)"
```

Kết quả mong muốn:

```text
[CHUNKER] Created ... semantic chunks.
START BUILD QDRANT...
[QDRANT] Đã build collection: thu_tuc_hanh_chinh_lai_chau
[QDRANT] Tổng số chunks: ...
BUILD DONE: ...
```

## 9. Test Qdrant search trực tiếp

Tạo file `test_qdrant_search.py` trong thư mục `backend`:

```python
from rag.vectorstore import load_existing_vectorstore

query = "Thủ tục đăng ký khai sinh cần giấy tờ gì?"

db = load_existing_vectorstore()

if db is None:
    print("DB is None. Chưa load được Qdrant collection.")
    raise SystemExit

docs = db.similarity_search(query, k=3)

print("QUERY:", query)
print("DOCS:", len(docs))

for i, doc in enumerate(docs, start=1):
    print("\n" + "=" * 60)
    print("DOC", i)
    print("=" * 60)
    print(doc.page_content[:1000])
    print("\nMETA:")
    print(doc.metadata)
```

Chạy test:

```powershell
python test_qdrant_search.py
```

## 10. Chạy backend FastAPI

Trước khi chạy backend, cần đảm bảo Qdrant đã chạy. Thứ tự khởi động chuẩn là:

```text
1. Mở Docker Desktop
2. Chạy hoặc kiểm tra container qdrant_local
3. Kích hoạt venv của backend
4. Chạy FastAPI bằng Uvicorn
```

Lệnh chạy đầy đủ:

```powershell
docker start qdrant_local
cd D:\.DA\backend
.\venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Nếu Qdrant đã được đặt restart policy bằng `docker update --restart unless-stopped qdrant_local` và Docker Desktop đang chạy, có thể bỏ qua lệnh `docker start qdrant_local`.

Mở Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Log khởi động mong muốn:

```text
=== STARTING RAG SYSTEM ===
[QDRANT] Đã load collection: thu_tuc_hanh_chinh_lai_chau
=== ĐÃ LOAD VECTOR DB CÓ SẴN (BỎ QUA CHUNKING) ===
=== SYSTEM READY ===
Application startup complete.
```

Nếu thấy `[QDRANT] Chưa có collection` hoặc backend không load được vector database, cần kiểm tra lại:

```powershell
docker ps
Invoke-RestMethod http://localhost:6333/collections
```


## 11. Test API

### 11.1. Test tìm kiếm thủ tục public

Endpoint:

```text
GET /user/procedures/search?q=khai sinh
```

Kết quả mong muốn: trả về danh sách thủ tục liên quan đến khai sinh.

### 11.2. Test chat local không cần đăng nhập

Chỉ dùng khi `.env` có:

```env
ENABLE_DEV_ROUTES=true
```

Endpoint:

```text
GET /dev/chat?q=Thủ tục đăng ký khai sinh cần giấy tờ gì?&session_id=dev_qdrant_test
```

Kết quả mong muốn:

```json
{
  "ok": true,
  "query": "Thủ tục đăng ký khai sinh cần giấy tờ gì?",
  "session_id": "dev_qdrant_test",
  "answer": "..."
}
```

### 11.3. Test chat thật có đăng nhập

Endpoint:

```text
GET /user/chat?q=Thủ tục đăng ký khai sinh cần giấy tờ gì?&session_id=user_qdrant_test
```

API này yêu cầu Firebase token:

```text
Authorization: Bearer <firebase_id_token>
```

Có thể lấy token từ frontend sau khi đăng nhập, trong tab Network của trình duyệt.

## 12. Kiến trúc RAG Pipeline hiện tại

1. Người dùng gửi câu hỏi.
2. Hệ thống kiểm tra intent xã giao.
3. LLM nhẹ rewrite câu hỏi để tối ưu truy vấn.
4. Chuẩn hóa câu hỏi bằng `synonyms.json`.
5. Nhận diện tên thủ tục bằng `entities.json`.
6. Nhận diện field cần hỏi, ví dụ: hồ sơ, phí, lệ phí, thời hạn, cơ quan, căn cứ pháp lý.
7. Truy xuất semantic từ Qdrant.
8. Nếu nhận diện được thủ tục, truy xuất thêm theo filter Qdrant `metadata.name`.
9. Rerank bằng CrossEncoder.
10. Nếu đã nhận diện được thủ tục rõ ràng, ưu tiên thủ tục đó làm winner.
11. Nếu câu hỏi có field cụ thể, exact field boost lấy trực tiếp chunk theo `metadata.name` và `metadata.field`.
12. Tổng hợp context tối đa `MAX_CONTEXT_CHUNKS`.
13. Gemini sinh câu trả lời dựa trên context.
14. Backend lưu lịch sử chat vào Firestore.

## 13. Các endpoint chính

### User

```text
GET /user/chat
GET /user/sessions
GET /user/chat/history
GET /user/procedures/search
GET /user/procedures/{procedure_id}
```

### Auth

```text
POST /auth/send-otp
POST /auth/verify-otp
GET  /auth/check-admin
```

### Admin

```text
POST /admin/reload-vectordb
GET  /admin/stats
GET  /admin/users
GET  /admin/users/{uid}/sessions
GET  /admin/history/sessions/{session_id}
DELETE /admin/history/sessions/{session_id}
```

### Dev local

```text
GET /dev/chat
```

Chỉ bật khi:

```env
ENABLE_DEV_ROUTES=true
```

## 14. Lưu ý khi deploy

Trước khi deploy hoặc public backend, kiểm tra:

```env
ENABLE_DEV_ROUTES=false
```

Không public các file nhạy cảm:

```text
.env
serviceAccountKey.json
venv/
__pycache__/
```

Nếu deploy lên cloud, cần dùng Qdrant Cloud hoặc một Qdrant server riêng có URL public/private phù hợp, sau đó đổi:

```env
QDRANT_URL=https://your-qdrant-url
```

Nếu vẫn chạy Qdrant local thì backend cloud sẽ không truy cập được `localhost:6333` trên máy cá nhân.

## 15. Ngrok cho backend local

Khi frontend đã deploy trên Firebase Hosting nhưng backend vẫn chạy local, có thể dùng Ngrok để public backend tạm thời.

Chạy backend:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Chạy Ngrok:

```powershell
ngrok http 8000
```

Ngrok sẽ cấp URL dạng:

```text
https://xxxx.ngrok-free.app
```

Cập nhật URL này vào frontend để frontend gọi backend local.

## 16. Ghi chú về backup vector database

Các endpoint backup cũ trong admin được thiết kế cho ChromaDB legacy. Sau khi chuyển sang Qdrant local, nên ưu tiên backup bằng cơ chế của Qdrant, Docker volume hoặc snapshot. Nếu cần dùng backup/restore trực tiếp trên dashboard admin, cần cập nhật riêng nhóm API `/admin/backups` theo Qdrant.

## 17. Lệnh thường dùng

### 17.1. Khởi động lại toàn bộ backend sau khi mở máy

Dùng bộ lệnh này khi bố mở máy và muốn chạy lại hệ thống backend local:

```powershell
docker start qdrant_local
cd D:\.DA\backend
.\venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Sau đó mở:

```text
http://127.0.0.1:8000/docs
```

### 17.2. Kiểm tra Qdrant

```powershell
docker ps
Invoke-RestMethod http://localhost:6333
Invoke-RestMethod http://localhost:6333/collections
```

Mở dashboard:

```text
http://localhost:6333/dashboard
```

### 17.3. Khởi động Qdrant nếu đã có container

```powershell
docker start qdrant_local
```

### 17.4. Dừng Qdrant khi không dùng

```powershell
docker stop qdrant_local
```

### 17.5. Cho Qdrant tự chạy lại cùng Docker

```powershell
docker update --restart unless-stopped qdrant_local
```

### 17.6. Chạy Qdrant bằng Docker Compose

Nếu dùng `docker-compose.yml` trong thư mục backend:

```powershell
cd D:\.DA\backend
docker compose up -d
```

Dừng Compose:

```powershell
docker compose down
```

Lưu ý: `docker compose down` có thể dừng container Compose. Không dùng tùy chọn xóa volume nếu chưa backup dữ liệu vector.

### 17.7. Chạy backend FastAPI

```powershell
cd D:\.DA\backend
.\venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 17.8. Bật hoặc tắt route dev

Bật route test local:

```env
ENABLE_DEV_ROUTES=true
```

Tắt route test local trước khi deploy hoặc demo public:

```env
ENABLE_DEV_ROUTES=false
```

### 17.9. Test nhanh thư viện backend

```powershell
python -c "from qdrant_client import QdrantClient; from langchain_qdrant import QdrantVectorStore; from langchain_huggingface import HuggingFaceEmbeddings; from rapidfuzz import fuzz; print('Libraries OK')"
```

