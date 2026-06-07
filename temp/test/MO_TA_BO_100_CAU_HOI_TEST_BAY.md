# Bộ 100 câu hỏi kiểm thử chatbot RAG/Qdrant có bẫy

Bộ test này được tạo dựa trên dữ liệu thủ tục trong `procedures.json`.

## Cấu trúc bộ test

Tổng số câu: **100**

- 60 câu thường: hỏi hồ sơ, thời hạn, phí/lệ phí, cơ quan thực hiện, kết quả.
- 25 câu bẫy: giả định sai, nhầm thủ tục gần nghĩa, nhầm phí/thời hạn, prompt injection, ngoài phạm vi.
- 15 câu theo ngữ cảnh hội thoại: dùng `previous_context` để kiểm tra khả năng nhớ phiên và đổi thủ tục.

## Các nhóm bẫy chính

- `false_time`: gài thời hạn sai.
- `false_fee`: gài phí/lệ phí sai.
- `wrong_agency`: gài cơ quan sai.
- `wrong_document`: gài giấy tờ thuộc thủ tục khác.
- `fake_calculation`: lấy ngày ban hành văn bản để tính thời hạn.
- `confusing_similar_procedure`: cố tình nhầm thủ tục gần nghĩa.
- `context_switch`: đang hỏi thủ tục này rồi đổi sang thủ tục khác.
- `instruction_attack`: yêu cầu bỏ qua dữ liệu và bịa thông tin.
- `out_of_scope`: hỏi thủ tục không có trong dữ liệu.

## Cột quan trọng

- `test_id`: mã câu test.
- `test_group`: normal, trap hoặc context.
- `difficulty`: easy, medium, hard.
- `question`: câu hỏi gửi vào chatbot.
- `previous_context`: câu hỏi/ngữ cảnh trước đó nếu có.
- `expected_procedure_id`: mã thủ tục kỳ vọng.
- `expected_procedure_name`: tên thủ tục kỳ vọng.
- `expected_field`: nhóm thông tin cần trả lời.
- `expected_behavior`: cách chatbot nên xử lý.
- `expected_keywords`: từ khóa/gợi ý đối chiếu khi chấm kết quả.

## Cách chạy với script test

Đặt file CSV cùng thư mục với `run_chatbot_tests.py`, sau đó chạy:

```powershell
python run_chatbot_tests.py --input Bo_100_cau_hoi_test_chatbot_RAG_bay.csv --output ket_qua_test_chatbot_RAG_bay.csv
```

Nếu test qua route dev:

```powershell
python run_chatbot_tests.py --input Bo_100_cau_hoi_test_chatbot_RAG_bay.csv --output ket_qua_test_chatbot_RAG_bay.csv --endpoint dev
```

## Lưu ý khi chấm

Cột `expected_keywords` không phải đáp án tuyệt đối, mà là gợi ý để đối chiếu nhanh.
Các câu bẫy cần đọc thêm `expected_behavior` để biết chatbot có phản biện đúng giả định sai hay không.
