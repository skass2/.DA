"""
Chạy bộ 100 câu hỏi kiểm thử chatbot RAG.

Cách dùng:
  pip install requests
  
  # Chạy trực tiếp bằng cách truyền token vào lệnh:
  python run_chatbot_tests.py --token "ĐIỀN_TOKEN_VÀO_ĐÂY"

Ghi chú:
- Mặc định script sẽ gọi tới http://localhost:8000. Nếu cổng khác, dùng thêm: --base-url http://localhost:xxxx
- Mặc định dùng GET /user/chat. Nếu dùng POST thì thêm tham số: --method POST
- File kết quả mặc định lưu vào: ket_qua_test_chatbot_RAG.csv
"""
import argparse
import csv
import os
import time
from datetime import datetime
import requests
from urllib.parse import quote_plus

# ĐIỀN TOKEN CỦA BẠN VÀO ĐÂY ĐỂ CHẠY TRỰC TIẾP BẰNG NÚT PLAY TRONG IDE:
MY_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImVlYzIxN2Q0MThjYjhlNWEzMTQzMThhMGQyZmZhNGUwY2ViMmU0Y2MiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiVmluaCBOZ3V54buFbiIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NJUzFVVWp4a3FlSVhYZExEeVdZbzkxalpsdExnbVNFTWVqQl9DNUNhUWVvdjZOME9yND1zOTYtYyIsImlzcyI6Imh0dHBzOi8vc2VjdXJldG9rZW4uZ29vZ2xlLmNvbS9teWNoYXRib3QtNzAyMSIsImF1ZCI6Im15Y2hhdGJvdC03MDIxIiwiYXV0aF90aW1lIjoxNzc4NDk5Mjg4LCJ1c2VyX2lkIjoid0hFdHh0cVYxalc1RUdSZ0tXY1BsT2JsMzdRMiIsInN1YiI6IndIRXR4dHFWMWpXNUVHUmdLV2NQbE9ibDM3UTIiLCJpYXQiOjE3NzkzOTE2OTcsImV4cCI6MTc3OTM5NTI5NywiZW1haWwiOiJuZ3Zpbmg3MDIxQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7Imdvb2dsZS5jb20iOlsiMTE3NjIxMzYzOTE5Mjk2MjAzMjYyIl0sImVtYWlsIjpbIm5ndmluaDcwMjFAZ21haWwuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoiZ29vZ2xlLmNvbSJ9fQ.XEEPWjiDr-sYWlgYSWcVW-iYHhevpH5toYuz1Br4rufTRRUmZnMVyuj2pzGvoiQK-giwY1EeUKg73JH2hmF9tx7eIxumFsf9EPx0VA0mEr47cEIYYV9VNZs5-6Sm9WyjH6m5IDwARKKfXVtZvnWwaip2ORXdlRF-afNIxAXPFtb_ZWZC7dAEv-8A07_39P_MeM4a0uzZZGrKL-Vd45ZDarT4mBQVP5NEYekjOWKIZWU9H7_uAHM7vzGQ21rAqbsD60zLr7Z3tewyN4r0rS45Um11HdUM24ZeN1arB6W5WSual32rlV65tu7ykCN3Kx82XDviMXWbOXn3qLOpm0CjHg"


def run_case(base_url, token, row, method='GET'):
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    session_id = f"test_{row['test_id']}"
    if row.get('previous_context'):
        # Gửi câu trước để tạo ngữ cảnh hội thoại
        _send(base_url, headers, row['previous_context'], session_id, method)
    start = time.perf_counter()
    status_code, body = _send(base_url, headers, row['question'], session_id, method)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return status_code, body, latency_ms


def _send(base_url, headers, question, session_id, method):
    url = base_url.rstrip('/') + '/user/chat'
    if method.upper() == 'POST':
        res = requests.post(url, headers=headers, json={'q': question, 'session_id': session_id}, timeout=60)
    else:
        res = requests.get(url, headers=headers, params={'q': question, 'session_id': session_id}, timeout=60)
    try:
        body = res.json()
    except Exception:
        body = {'raw_text': res.text}
    return res.status_code, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='Bo_100_cau_hoi_test_chatbot_RAG.csv')
    parser.add_argument('--output', default='ket_qua_test_chatbot_RAG.csv')
    parser.add_argument('--base-url', default=os.getenv('BASE_URL', 'http://localhost:8000'))
    parser.add_argument('--token', default=os.getenv('FIREBASE_ID_TOKEN', MY_TOKEN))
    parser.add_argument('--method', default='GET', choices=['GET', 'POST'])
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    # Các cột xuất ra theo đúng yêu cầu của bạn
    out_fields = [
        'test_id', 'thoi_gian_test', 'cau_hoi', 'cau_tra_loi_mong_muon',
        'cau_tra_loi_nhan_duoc', 'thoi_gian_phan_hoi_ms', 'danh_gia'
    ]

    # Bước 1: Đọc file kết quả cũ để tìm xem đã test đến câu nào rồi
    processed_ids = set()
    latencies = []
    file_exists = os.path.exists(args.output)
    if file_exists:
        with open(args.output, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('test_id'):
                    processed_ids.add(row['test_id'])
                    if row.get('thoi_gian_phan_hoi_ms') and row['thoi_gian_phan_hoi_ms'].isdigit():
                        latencies.append(int(row['thoi_gian_phan_hoi_ms']))

    print(f"[*] Đã tìm thấy {len(processed_ids)} câu hỏi đã được test. Sẽ tiếp tục với các câu còn lại...")

    # Bước 2: Mở file ở chế độ 'a' (append) để nối thêm dữ liệu
    try:
        with open(args.output, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            if not file_exists:
                writer.writeheader()
                
            for row in rows:
                test_id = row.get('test_id')
                if test_id in processed_ids:
                    continue # Bỏ qua nếu câu này đã có kết quả
                    
                print(f"[*] Đang test câu {test_id}...")
                try:
                    status, body, latency = run_case(args.base_url, args.token, row, args.method)
                    answer = body.get('answer') if isinstance(body, dict) else str(body)
                    
                    # Nếu API trả về lỗi 429 (Quá giới hạn) hoặc các lỗi server, ngắt ngang luôn
                    if status != 200:
                        print(f"[!] Lỗi HTTP {status}. Có thể bạn đã hết lượt free trial. Dừng tiến trình để bảo toàn vị trí.")
                        break
                        
                    out = {
                        'test_id': test_id,
                        'thoi_gian_test': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'cau_hoi': row.get('question', ''),
                        'cau_tra_loi_mong_muon': row.get('expected_behavior', ''),
                        'cau_tra_loi_nhan_duoc': answer,
                        'thoi_gian_phan_hoi_ms': latency,
                        'danh_gia': '' # Cột này để trống cho bạn tự điền sau
                    }
                except Exception as e:
                    print(f"[!] Gặp sự cố kết nối: {e}. Dừng tiến trình để kiểm tra lại hệ thống.")
                    break # Ngắt luôn vòng lặp, không ghi bậy vào file
                    
                writer.writerow(out)
                f.flush() # ÉP GHI XUỐNG Ổ CỨNG NGAY LẬP TỨC
                latencies.append(latency)
                print(f"[+] Hoàn thành câu {test_id} - Phản hồi: {latency}ms")
                
                # Delay nhẹ 2s để tránh bị AI đánh chặn vì spam request quá nhanh
                time.sleep(2)
                
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"\n[=] TỔNG KẾT HIỆU NĂNG:")
            print(f"    - Đã có kết quả: {len(latencies)}/{len(rows)} câu")
            print(f"    - Thời gian phản hồi trung bình: {avg_latency:.0f} ms (khoảng {avg_latency/1000:.2f} giây)\n")
    except PermissionError:
        print(f"\n[!] LỖI: Không thể ghi vào file '{args.output}'.")
        print("[!] Nguyên nhân: Có thể bạn đang mở file này bằng Excel hoặc một phần mềm khác.")
        print("[!] Cách khắc phục: Vui lòng đóng file Excel lại và chạy lại lệnh.\n")


if __name__ == '__main__':
    main()
