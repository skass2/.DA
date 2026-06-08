# RAG Chatbot Tra Cứu Thủ Tục Hành Chính

## Demo Web

https://mychatbot-7021.web.app/

## 1. Giới thiệu

Dự án xây dựng chatbot hỗ trợ tra cứu thủ tục hành chính bằng mô hình RAG (Retrieval-Augmented Generation). Hệ thống cho phép người dùng hỏi bằng ngôn ngữ tự nhiên để tra cứu hồ sơ, lệ phí, thời hạn, cơ quan giải quyết, kết quả, căn cứ pháp lý và các thông tin liên quan đến thủ tục hành chính.

Phiên bản hiện tại sử dụng Qdrant local chạy bằng Docker làm vector database. Dữ liệu gốc vẫn được quản lý bằng JSON, sau đó được chunking thành nội dung dạng Markdown/Text rõ ngữ cảnh để embedding và truy xuất trong Qdrant.

Mục tiêu chính của hệ thống là tạo một chatbot dễ dùng, trả lời ngắn gọn, thân thiện với người dân, hạn chế đoán sai thủ tục và chỉ gọi RAG/LLM khi thật sự cần.

## 2. Tính năng chính

- Tra cứu thủ tục hành chính bằng câu hỏi tự nhiên.
- Tìm kiếm thủ tục theo tên, lĩnh vực hoặc cụm từ liên quan.
- Hỏi đáp theo từng nhóm thông tin:
  - Thành phần hồ sơ
  - Thời hạn giải quyết
  - Phí/lệ phí
  - Cơ quan thực hiện
  - Kết quả thực hiện
  - Căn cứ pháp lý
- Qdrant local dùng làm vector database.
- Chunking theo hướng contextual/Markdown:
  - JSON là dữ liệu gốc.
  - Markdown/Text là nội dung dùng để embedding và đưa vào LLM.
  - Mỗi chunk có metadata như tên thủ tục, mã thủ tục, field, section type.
- Có parent context cho toàn bộ thủ tục và child chunk theo từng mục thông tin.
- Lưu thủ tục đang tư vấn theo session để xử lý câu hỏi nối tiếp.
- Exact field retrieval:
  - Khi đã biết thủ tục và field cần hỏi, backend lấy trực tiếp chunk theo metadata.
  - Giảm việc search toàn bộ vector database.
- Conversation-aware query router:
  - Nhận diện câu chào hỏi, cảm ơn, xác nhận đã hiểu.
  - Nhận diện câu nối tiếp đang bám thủ tục hiện tại.
  - Nhận diện câu chuyển sang thủ tục khác.
  - Nhận diện câu hỏi yêu cầu danh sách thủ tục.
  - Hỏi lại khi thủ tục chưa rõ, thay vì đoán bừa.
  - Không gọi Qdrant/LLM khi câu hỏi không đủ bằng chứng thuộc miền thủ tục hành chính.
- Robust procedure resolver:
  - Giảm nhầm giữa các nhóm dễ lẫn như đăng ký, cấp bản sao, cấp lại, sửa đổi, thông báo.
  - So khớp theo hành động, đối tượng chính và bổ nghĩa thay vì chỉ dựa vào substring.
- List query route:
  - Hỗ trợ câu hỏi như “cho tôi danh sách các thủ tục kết hôn”, “có những thủ tục nào liên quan đến đất đai”.
  - Trả về danh sách thủ tục phù hợp, không chọn một thủ tục đơn lẻ.
  - Dùng token/phrase matching chặt để tránh khớp sai chuỗi con.
- Direct answer formatter:
  - Hỏi hồ sơ chỉ trả hồ sơ.
  - Hỏi thời hạn chỉ trả thời hạn.
  - Hỏi phí/lệ phí chỉ trả phí/lệ phí.
  - Hỏi nơi nộp chỉ trả cơ quan/nơi giải quyết.
- Gợi ý câu hỏi tiếp theo khi đã xác định chắc thủ tục.
- Không hiển thị gợi ý và metadata với câu chào hỏi, xác nhận, mơ hồ hoặc ngoài phạm vi.
- API `/user/chat` trả response có cấu trúc:
  - `answer`
  - `suggested_questions`
  - `selected_procedure`
  - `sources`
  - `procedure_candidates`
  - `show_metadata`
- Frontend hỗ trợ:
  - Gợi ý câu hỏi dạng chip.
  - Metadata thủ tục/nguồn dạng icon/tooltip gọn.
  - Chat theo phiên.
- Admin hỗ trợ:
  - Quản lý tài liệu thủ tục.
  - Xem/sửa nội dung dạng Markdown hoặc JSON.
  - Tự rebuild Qdrant sau khi thêm, sửa hoặc xóa tài liệu.
- Endpoint dev local `/dev/chat` để test nhanh RAG không cần đăng nhập, chỉ bật khi `ENABLE_DEV_ROUTES=true`.

## 3. Công nghệ sử dụng

### Backend

- Python 3.10
- FastAPI
- Uvicorn
- LangChain
- SentenceTransformers
- Qdrant Client
- LangChain Qdrant
- LangChain HuggingFace
- Firebase Admin SDK
- Python Dotenv
- RapidFuzz

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

ENABLE_QUERY_REWRITE=auto
ENABLE_DIRECT_ANSWER=true
USE_LIGHTWEIGHT_FOR_ANSWER=true

MAX_CONTEXT_CHUNKS=7
DIRECT_ANSWER_MAX_CHUNKS=4
STRICT_FIELD_CONTEXT=true

RETRIEVAL_SEMANTIC_K=8
FILTER_SEARCH_K=5
PARENT_METHOD_K=2
EXACT_FIELD_K=6

LLM_RETRY_ATTEMPTS=1
LLM_RETRY_BASE_DELAY=0
LLM_SLEEP_ON_429=false
```

Ghi chú:

- Khi test local nhanh bằng `/dev/chat`, có thể đặt `ENABLE_DEV_ROUTES=true`.
- Khi deploy hoặc demo public, nên đặt `ENABLE_DEV_ROUTES=false` để tắt route test không cần đăng nhập.
- Không commit `.env` và `serviceAccountKey.json` lên GitHub.
- Có thể giảm `MAX_CONTEXT_CHUNKS` hoặc `DIRECT_ANSWER_MAX_CHUNKS` nếu muốn câu trả lời ngắn hơn.
- `ENABLE_QUERY_REWRITE=auto` giúp giảm số lần gọi LLM rewrite không cần thiết.

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

## 7. Cài đặt frontend

Vào thư mục frontend:

```powershell
cd D:\.DA\chatbot
npm install
npm run dev
```

Build kiểm tra:

```powershell
npm run build
```

## 8. Cài đặt và chạy Qdrant local bằng Docker

Qdrant là vector database dùng để lưu embedding của dữ liệu thủ tục hành chính. Backend hiện tại kết nối tới Qdrant qua địa chỉ:

```env
QDRANT_URL=http://localhost:6333
```

Trước khi chạy backend FastAPI, cần đảm bảo Docker Desktop đã mở và container Qdrant đang chạy.

### 8.1. Kiểm tra Docker

Mở Docker Desktop và đợi Docker Engine chạy xong. Sau đó kiểm tra bằng PowerShell:

```powershell
docker ps
```

Nếu lệnh không báo lỗi thì Docker đã sẵn sàng.

### 8.2. Chạy Qdrant lần đầu bằng Docker

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

### 8.3. Khởi động lại Qdrant khi container đã tồn tại

Nếu container `qdrant_local` đã được tạo từ trước, mỗi lần mở lại máy chỉ cần chạy:

```powershell
docker start qdrant_local
```

Kiểm tra container đang chạy:

```powershell
docker ps
```

Nếu thấy dòng có tên `qdrant_local` và cổng `6333-6334` thì Qdrant đã chạy.

### 8.4. Dùng docker-compose.yml nếu muốn quản lý Qdrant bằng Compose

Nếu thư mục backend có file `docker-compose.yml`, có thể dùng Docker Compose để chạy Qdrant:

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

### 8.5. Cho Qdrant tự khởi động cùng Docker

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

## 9. Nạp dữ liệu thủ tục vào Qdrant

Dữ liệu gốc nằm trong:

```text
backend/data/procedures.json
```

Quy trình build vector database:

```text
procedures.json
-> loader.py
-> chunker.py
-> Markdown/contextual chunks
-> embedding model
-> Qdrant collection
```

Sau khi Qdrant chạy, build vector database bằng lệnh:

```powershell
cd D:\.DA\backend
.\venv\Scripts\activate

python -c "from rag.loader import load_data; from rag.chunker import create_chunks; from rag.vectorstore import build_vectorstore; data=load_data(); chunks=create_chunks(data); print('START BUILD QDRANT...'); db=build_vectorstore(chunks, backup=True); print('BUILD DONE:', db)"
```

Kết quả mong muốn:

```text
[CHUNKER] Created ... semantic/contextual chunks.
START BUILD QDRANT...
[QDRANT] Đã build collection: thu_tuc_hanh_chinh_lai_chau
[QDRANT] Tổng số chunks: ...
BUILD DONE: ...
```

## 10. Đồng bộ dữ liệu từ Admin

Trang admin quản lý dữ liệu thủ tục trong `procedures.json`.

Luồng xử lý khi thêm, sửa hoặc xóa tài liệu:

```text
Admin thêm/sửa/xóa tài liệu
-> Lưu lại procedures.json
-> Chạy chunker.py
-> Tạo lại embeddings
-> Rebuild Qdrant
-> Backend dùng collection mới
```

Lưu ý:

- Quá trình rebuild Qdrant có thể mất vài phút nếu dữ liệu nhiều.
- Trong lúc rebuild, request admin có thể chờ lâu nếu backend đang xử lý đồng bộ trực tiếp.
- Nếu cần tối ưu trải nghiệm admin, có thể chuyển rebuild sang background task kèm endpoint kiểm tra tiến trình.
- Khi chỉ sửa `pipeline.py`, `user.py`, frontend hoặc logic router, không cần build lại Qdrant.

Endpoint reload thủ công:

```text
POST /admin/reload-vectordb
```

## 11. Test Qdrant search trực tiếp

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

## 12. Chạy backend FastAPI

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

## 13. Lệnh khởi động toàn bộ backend trong một lần

Có thể chạy một lệnh PowerShell để mở Docker Desktop nếu cần, khởi động Qdrant, kích hoạt venv và chạy FastAPI:

```powershell
powershell -NoExit -ExecutionPolicy Bypass -Command "cd 'D:\.DA\backend'; if (-not (docker info 2>$null)) { Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'; Write-Host 'Dang cho Docker khoi dong...'; do { Start-Sleep -Seconds 2 } until (docker info 2>$null) }; docker start qdrant_local | Out-Host; if (Test-Path '.\.venv\Scripts\Activate.ps1') { . .\.venv\Scripts\Activate.ps1 } elseif (Test-Path '.\venv\Scripts\Activate.ps1') { . .\venv\Scripts\Activate.ps1 } else { Write-Host 'Khong tim thay moi truong ao .venv hoac venv'; exit 1 }; uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
```

## 14. Test API

### 14.1. Test tìm kiếm thủ tục public

Endpoint:

```text
GET /user/procedures/search?q=khai sinh
```

Kết quả mong muốn: trả về danh sách thủ tục liên quan đến khai sinh.

### 14.2. Test chat local không cần đăng nhập

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

### 14.3. Test chat thật có đăng nhập

Endpoint:

```text
GET /user/chat?q=Thủ tục đăng ký khai sinh cần giấy tờ gì?&session_id=user_qdrant_test
```

API này yêu cầu Firebase token:

```text
Authorization: Bearer <firebase_id_token>
```

Có thể lấy token từ frontend sau khi đăng nhập, trong tab Network của trình duyệt.

### 14.4. Response mẫu của `/user/chat`

```json
{
  "answer": "Thời hạn giải quyết thủ tục đăng ký khai sinh là trong ngày tiếp nhận hồ sơ.",
  "suggested_questions": [
    "Hồ sơ cần chuẩn bị cho thủ tục đăng ký khai sinh gồm những gì?",
    "Thủ tục đăng ký khai sinh có mất phí hoặc lệ phí không?",
    "Tôi cần nộp thủ tục đăng ký khai sinh ở đâu?"
  ],
  "selected_procedure": {
    "id": "1.001193.000.00.00.H35",
    "name": "Thủ tục đăng ký khai sinh",
    "display_name": "đăng ký khai sinh"
  },
  "sources": [
    {
      "procedure_id": "1.001193.000.00.00.H35",
      "procedure_name": "Thủ tục đăng ký khai sinh",
      "field": "Thời hạn giải quyết",
      "section_type": "method",
      "chunk_id": "..."
    }
  ],
  "procedure_candidates": [],
  "show_metadata": true
}
```

## 15. Kiến trúc RAG Pipeline hiện tại

Luồng xử lý chính:

```text
1. Người dùng gửi câu hỏi.
2. Query router phân loại:
   - ACKNOWLEDGEMENT
   - SOCIAL
   - CONTINUE_CONTEXT
   - SWITCH_PROCEDURE
   - ADMIN_CONFIDENT
   - ASK_CLARIFY
   - LIST_QUERY
   - OUT_OF_DOMAIN
3. Nếu là câu xã giao, xác nhận, ngoài phạm vi hoặc hỏi chưa rõ thủ tục:
   - Trả lời nhanh.
   - Không gọi Qdrant.
   - Không gọi LLM.
   - Không hiển thị metadata/nguồn.
4. Nếu là câu hỏi danh sách:
   - Lọc danh sách thủ tục từ procedures.json bằng token/phrase matching.
   - Không chọn một thủ tục đơn lẻ.
5. Nếu đã có thủ tục đang chọn và câu hỏi là nối tiếp:
   - Dùng selected procedure context.
   - Không search toàn bộ database.
6. Nếu câu hỏi có thủ tục mới rõ ràng:
   - Resolver xác định thủ tục mới.
   - Cập nhật selected procedure theo session.
7. Nhận diện field cần hỏi:
   - Hồ sơ
   - Phí/lệ phí
   - Thời hạn
   - Cơ quan
   - Kết quả
   - Căn cứ pháp lý
8. Nếu đã biết thủ tục và field:
   - Exact field retrieval bằng Qdrant scroll/filter.
   - Có thể trả lời trực tiếp bằng formatter.
9. Nếu chưa đủ chắc:
   - Semantic retrieval từ Qdrant.
   - Rerank bằng CrossEncoder.
   - Chọn context phù hợp.
10. Gemini sinh câu trả lời khi cần diễn giải sâu hoặc xử lý câu hỏi bẫy.
11. Backend lưu lịch sử chat vào Firestore.
```

Nguyên tắc hoạt động:

```text
Không chắc thì hỏi lại.
Có dấu hiệu nối tiếp thì không cắt mạch hội thoại.
Có thủ tục đang chọn thì ưu tiên giữ mạch.
Có thủ tục mới rõ ràng thì chuyển mạch.
Câu lệch hẳn miền thủ tục hành chính thì dừng sớm.
Không sinh gợi ý khi chưa chắc thủ tục.
Không dùng context cũ để kéo câu hỏi ngoài phạm vi vào RAG.
```

## 16. Query Router

Query router được đặt trước tầng retrieval để giảm nhầm thủ tục và tiết kiệm tài nguyên.

### 16.1. ACKNOWLEDGEMENT

Ví dụ:

```text
à
ừ tôi biết rồi
tôi hiểu rồi cảm ơn
```

Hành vi:

```text
Trả lời ngắn.
Không gọi Qdrant.
Không gọi LLM.
Không gợi ý câu hỏi.
Không hiện metadata.
```

### 16.2. SOCIAL

Ví dụ:

```text
xin chào
cảm ơn
tạm biệt
```

Hành vi tương tự ACKNOWLEDGEMENT.

### 16.3. CONTINUE_CONTEXT

Ví dụ:

```text
thế phí thì sao
nộp ở đâu
bao lâu có kết quả
tôi chưa biết phải làm thế nào
```

Hành vi:

```text
Nếu session đã có selected procedure, tiếp tục trong phạm vi thủ tục đó.
Không search toàn bộ database nếu field đã rõ.
```

### 16.4. SWITCH_PROCEDURE

Ví dụ:

```text
thế đăng ký kết hôn thì sao
quay lại đăng ký khai sinh
không phải bản sao, tôi muốn đăng ký khai sinh lần đầu
```

Hành vi:

```text
Resolver xác định thủ tục mới.
Nếu đủ chắc, cập nhật selected procedure.
Nếu chưa chắc, hỏi lại người dùng.
```

### 16.5. ASK_CLARIFY

Ví dụ:

```text
tôi muốn làm giấy cho con
tôi bị mất giấy rồi
cấp lại cái đó được không
```

Hành vi:

```text
Nếu không đủ bằng chứng để chọn đúng thủ tục, hệ thống hỏi lại nhẹ nhàng.
Không đoán bừa.
```

### 16.6. LIST_QUERY

Ví dụ:

```text
cho tôi danh sách các thủ tục kết hôn
có những thủ tục nào liên quan đến đất đai
liệt kê thủ tục hộ tịch
```

Hành vi:

```text
Trả về danh sách thủ tục liên quan.
Không chọn một thủ tục đơn lẻ.
Không dùng selected context cũ.
```

### 16.7. OUT_OF_DOMAIN

Ví dụ:

```text
bạn có viết code được không
hôm nay thời tiết thế nào
```

Hành vi:

```text
Trả lời ngắn rằng hệ thống chỉ hỗ trợ tra cứu thủ tục hành chính.
Không gọi Qdrant.
Không gọi LLM.
Không gợi ý câu hỏi.
Không hiện metadata.
```

## 17. Robust Procedure Resolver

Resolver được thiết kế để giảm nhầm lẫn giữa các thủ tục có tên gần giống nhau.

Cơ chế chính:

```text
1. Tách nhóm hành động:
   - đăng ký
   - cấp bản sao/trích lục
   - cấp lại/cấp đổi
   - sửa đổi/bổ sung/điều chỉnh
   - thông báo/công bố
   - xác nhận/chứng nhận
   - cấp giấy phép

2. Tách đối tượng chính:
   - khai sinh
   - kết hôn
   - hộ kinh doanh
   - khuyến mại
   - đất đai
   - C/O
   - giấy phép
   - lĩnh vực khác

3. Tách bổ nghĩa:
   - lần đầu
   - bản sao
   - trích lục
   - bị mất
   - cấp lại
   - yếu tố nước ngoài
   - sửa đổi/bổ sung

4. Chấm điểm và phạt mismatch:
   - Câu hỏi “đăng ký giấy khai sinh lần đầu” không ưu tiên thủ tục cấp bản sao.
   - Câu hỏi “xin bản sao giấy khai sinh” không ưu tiên thủ tục đăng ký khai sinh lần đầu.
   - Câu hỏi “thông báo khuyến mại” không rơi vào “đăng ký sửa đổi chương trình khuyến mại”.
```

## 18. Chunking và Qdrant

### 18.1. Vai trò của `chunker.py`

`chunker.py` không phụ thuộc ChromaDB hay Qdrant. File này chỉ có nhiệm vụ chuyển dữ liệu JSON thành danh sách `Document`.

Luồng đúng:

```text
procedures.json
-> loader.py
-> chunker.py
-> Document(page_content=Markdown/Text, metadata=dict)
-> vectorstore.py
-> Qdrant
```

### 18.2. Vì sao dùng Markdown/Text cho chunk

Embedding model và LLM đọc chuỗi văn bản tốt hơn khi có ngữ cảnh rõ. Vì vậy mỗi chunk nên có heading, tên thủ tục, mã thủ tục, lĩnh vực và mục thông tin.

Ví dụ:

```md
# Thủ tục: Đăng ký khai sinh

- Mã thủ tục: 1.001193.000.00.00.H35
- Lĩnh vực: Hộ tịch
- Mục thông tin: Thành phần hồ sơ

## Thành phần hồ sơ

1. Giấy chứng sinh
   - Bản chính: 1
   - Bản sao: 0
```

### 18.3. Parent-child chunking

Hệ thống dùng hai nhóm chunk:

```text
Parent chunk:
- Toàn bộ thông tin của một thủ tục.
- Dùng khi người dùng hỏi sâu hoặc hỏi nối tiếp theo thủ tục đã chọn.

Child chunk:
- Hồ sơ
- Cách thức/thời hạn/phí
- Cơ quan/kết quả
- Điều kiện
- Căn cứ pháp lý
```

Cách này giúp giảm chunk rác, hạn chế lấy nhầm mẩu thông tin thiếu ngữ cảnh.

## 19. Các endpoint chính

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
GET  /admin/documents
POST /admin/documents
PUT  /admin/documents/{doc_id}
DELETE /admin/documents/{doc_id}
```

### Dev local

```text
GET /dev/chat
```

Chỉ bật khi:

```env
ENABLE_DEV_ROUTES=true
```

## 20. Lưu ý khi deploy

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

## 21. Ngrok cho backend local

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

## 22. Ghi chú về backup vector database

Các endpoint backup cũ trong admin được thiết kế cho ChromaDB legacy. Sau khi chuyển sang Qdrant local, nên ưu tiên backup bằng cơ chế của Qdrant, Docker volume hoặc snapshot. Nếu cần dùng backup/restore trực tiếp trên dashboard admin, cần cập nhật riêng nhóm API `/admin/backups` theo Qdrant.

## 23. Lệnh thường dùng

### 23.1. Khởi động lại toàn bộ backend sau khi mở máy

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

### 23.2. Kiểm tra Qdrant

```powershell
docker ps
Invoke-RestMethod http://localhost:6333
Invoke-RestMethod http://localhost:6333/collections
```

Mở dashboard:

```text
http://localhost:6333/dashboard
```

### 23.3. Khởi động Qdrant nếu đã có container

```powershell
docker start qdrant_local
```

### 23.4. Dừng Qdrant khi không dùng

```powershell
docker stop qdrant_local
```

### 23.5. Cho Qdrant tự chạy lại cùng Docker

```powershell
docker update --restart unless-stopped qdrant_local
```

### 23.6. Chạy Qdrant bằng Docker Compose

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

### 23.7. Chạy backend FastAPI

```powershell
cd D:\.DA\backend
.\venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 23.8. Bật hoặc tắt route dev

Bật route test local:

```env
ENABLE_DEV_ROUTES=true
```

Tắt route test local trước khi deploy hoặc demo public:

```env
ENABLE_DEV_ROUTES=false
```

### 23.9. Test nhanh thư viện backend

```powershell
python -c "from qdrant_client import QdrantClient; from langchain_qdrant import QdrantVectorStore; from langchain_huggingface import HuggingFaceEmbeddings; from rapidfuzz import fuzz; print('Libraries OK')"
```

### 23.10. Rebuild Qdrant thủ công

```powershell
cd D:\.DA\backend
.\venv\Scripts\activate
python -c "from rag.loader import load_data; from rag.chunker import create_chunks; from rag.vectorstore import build_vectorstore; data=load_data(); chunks=create_chunks(data); build_vectorstore(chunks, backup=True); print('DONE')"
```

## 24. Troubleshooting

### 24.1. Backend không load được Qdrant

Kiểm tra:

```powershell
docker ps
Invoke-RestMethod http://localhost:6333/collections
```

Nếu chưa có collection, cần build lại Qdrant từ `procedures.json`.

### 24.2. API chat trả lỗi 500 do numpy.float32

Nguyên nhân thường là score từ reranker chưa được convert sang kiểu JSON-safe.

Cách xử lý trong backend:

```text
Ép score sang float trước khi trả response.
Dùng make_json_safe() trong router user.py để xử lý numpy integer, numpy floating và ndarray.
```

### 24.3. Đồng bộ Qdrant bị đứng lâu trên admin

Nguyên nhân thường là hệ thống đang embedding nhiều chunk và rebuild collection.

Cách kiểm tra:

```powershell
Invoke-RestMethod http://localhost:6333/collections
Invoke-RestMethod http://localhost:6333/collections/thu_tuc_hanh_chinh_lai_chau
```

Nếu cần trải nghiệm tốt hơn, nên chuyển rebuild sang background task và thêm endpoint status.

### 24.4. Chatbot chọn nhầm thủ tục gần giống

Kiểm tra các lớp sau:

```text
detect_procedure_name()
robust procedure resolver
query router
selected procedure memory
field detection
```

Nếu người dùng yêu cầu danh sách, cần đảm bảo route `LIST_QUERY` được bắt trước khi resolver chọn một thủ tục đơn lẻ.

### 24.5. Câu hỏi ngoài phạm vi vẫn sinh gợi ý

Kiểm tra response từ pipeline:

```json
{
  "suggested_questions": [],
  "selected_procedure": null,
  "sources": [],
  "show_metadata": false
}
```

Frontend chỉ nên hiển thị chip gợi ý khi `suggested_questions` có dữ liệu.

## 25. Gợi ý commit tổng hợp

```bash
git add backend/requirements.txt \
        backend/rag/chunker.py \
        backend/rag/memory.py \
        backend/rag/pipeline.py \
        backend/rag/config.py \
        backend/routers/user.py \
        backend/routers/dev.py \
        backend/routers/admin.py \
        frontend/src/components/ChatBox.tsx \
        frontend/src/components/InputBox.tsx \
        frontend/src/components/Message.tsx \
        frontend/src/components/AdminDashboard.tsx \
        frontend/src/types/chat.ts \
        README.md

git commit -m "feat: improve Qdrant RAG context, routing, admin sync and chat UX"
```
