# RAG Chatbot Tra Cứu Thủ Tục Hành Chính

## Giới thiệu
Project xây dựng chatbot tra cứu thủ tục hành chính sử dụng mô hình RAG (Retrieval-Augmented Generation) kết hợp với Gemini.

---

## Tính năng
- File search bằng vector database
- Gemini 2.5 Flash / Pro để sinh câu trả lời
- MMR search (tăng độ đa dạng kết quả)
- Lọc theo metadata (field-based retrieval)
- Chunk dữ liệu theo từng trường
- Retry khi gọi LLM lỗi
- Reranking với Cross-Encoder
- Fallback search khi không có kết quả

---

## Kiến trúc
- User → Retriever → Filter → Rerank → Context → Gemini → Answer
