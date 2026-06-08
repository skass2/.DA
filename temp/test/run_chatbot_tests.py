"""
File: run_chatbot_tests.py
Mục đích:
- Chạy bộ câu hỏi kiểm thử chatbot RAG/Qdrant.
- Mặc định chỉ cần chạy: python run_chatbot_tests.py
- Hỗ trợ CSV và JSONL.
- Hỗ trợ /user/chat có Firebase token và /dev/chat không cần token.

Cách dùng nhanh:
1. Dán Firebase ID Token vào DEFAULT_FIREBASE_ID_TOKEN nếu muốn test /user/chat.
2. Đảm bảo backend đang chạy ở http://localhost:8000.
3. Đảm bảo file câu hỏi test nằm cùng thư mục với file này.
4. Chạy:
   python run_chatbot_tests.py

Lưu ý bảo mật:
- Nếu đã dán token thật vào DEFAULT_FIREBASE_ID_TOKEN thì KHÔNG commit file này lên GitHub.
- Token Firebase có hạn sử dụng. Nếu test bị 401, hãy lấy token mới từ frontend.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


# =========================================================
# CẤU HÌNH MẶC ĐỊNH
# =========================================================
# Bố có thể dán token Firebase vào đây để câu lệnh chạy ngắn nhất:
# python run_chatbot_tests.py
#
# Ví dụ:
# DEFAULT_FIREBASE_ID_TOKEN = "eyJhbGciOiJSUzI1NiIs..."
#
# Không commit file này nếu đã dán token thật.
DEFAULT_FIREBASE_ID_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6Ijg1NGFhNGMyM2VkZTdiOGNhODc1OWZiMDZlNmExZDU4OTI0MjVkMDYiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiVmluaCBOZ3V54buFbiIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NJUzFVVWp4a3FlSVhYZExEeVdZbzkxalpsdExnbVNFTWVqQl9DNUNhUWVvdjZOME9yND1zOTYtYyIsImlzcyI6Imh0dHBzOi8vc2VjdXJldG9rZW4uZ29vZ2xlLmNvbS9teWNoYXRib3QtNzAyMSIsImF1ZCI6Im15Y2hhdGJvdC03MDIxIiwiYXV0aF90aW1lIjoxNzgwODA0NjE5LCJ1c2VyX2lkIjoid0hFdHh0cVYxalc1RUdSZ0tXY1BsT2JsMzdRMiIsInN1YiI6IndIRXR4dHFWMWpXNUVHUmdLV2NQbE9ibDM3UTIiLCJpYXQiOjE3ODA4MDk0NzIsImV4cCI6MTc4MDgxMzA3MiwiZW1haWwiOiJuZ3Zpbmg3MDIxQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7Imdvb2dsZS5jb20iOlsiMTE3NjIxMzYzOTE5Mjk2MjAzMjYyIl0sImVtYWlsIjpbIm5ndmluaDcwMjFAZ21haWwuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoiZ29vZ2xlLmNvbSJ9fQ.TRWilCn8sEGxxB5JzTjPuihLE-bzwUuO6FqhqhI6T48kWehwsIE8cCAXoSVxE5pvf5a49SmlS3BzKzRPUY145GxFaVcQ9SJq5ELPUEdwp6aFxfOpF-J3lLjlDmBJZXAXj_MTlIMGCc-ufo78cuEEe5ulqP_5i3BvTWxZlOmydzVYUeuQcJScwjEWu4CA0NiefqyW0rbtP3Qt_99n-ckxReU5rR5G0_WV2aHOhV0rzhlbCPh6zWK0d9-OTQoo1sU7-8SM4Af7ibOJxaMto5X0UxKcVB_31Efgch-qPbRGfkeWdMT5jMMtUicjCWc3PTiquooBjDZcDI_NVa0RknhWMA"

# Backend FastAPI local
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# user = gọi /user/chat, cần Firebase token
# dev  = gọi /dev/chat, không cần token, cần ENABLE_DEV_ROUTES=true
DEFAULT_ENDPOINT = "user"

# Backend hiện tại đang dùng GET /user/chat?q=...&session_id=...
DEFAULT_METHOD = "GET"

# File input/output mặc định
DEFAULT_INPUT_FILE = "bo_15_cau_test_xoay_sau_nuoi_con_nuoi.csv"
DEFAULT_OUTPUT_FILE = "ket_qua_test_Bo_15_cau_test_xoay_sau_nuoi_con_nuoi.csv"

# Timeout mỗi câu hỏi, tính bằng giây
DEFAULT_TIMEOUT = 120

# Nghỉ giữa các request để giảm nguy cơ dính rate limit Gemini
DEFAULT_DELAY = 1.0


# =========================================================
# ĐỌC FILE INPUT
# =========================================================

def read_rows(input_path: str) -> List[Dict[str, str]]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file input: {input_path}. "
            f"Hãy đặt file test cùng thư mục hoặc truyền --input <duong_dan_file>."
        )

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows: List[Dict[str, str]] = []

        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Lỗi JSONL tại dòng {line_no}: {e}") from e

                rows.append({
                    str(k): "" if v is None else str(v)
                    for k, v in item.items()
                })

        return rows

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# =========================================================
# GỌI API
# =========================================================

def build_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")

    if endpoint == "dev":
        return f"{base}/dev/chat"

    return f"{base}/user/chat"


def build_headers(token: str, endpoint: str) -> Dict[str, str]:
    headers = {
        "accept": "application/json",
    }

    if endpoint == "user" and token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def parse_body(res: requests.Response) -> Dict[str, Any]:
    try:
        return res.json()
    except Exception:
        return {
            "raw_text": res.text,
        }


def extract_answer(body: Dict[str, Any]) -> str:
    if not isinstance(body, dict):
        return str(body)

    if "answer" in body:
        return str(body.get("answer") or "")

    if "detail" in body:
        return str(body.get("detail") or "")

    if "error" in body:
        return str(body.get("error") or "")

    return str(body)


def send_request(
    base_url: str,
    endpoint: str,
    headers: Dict[str, str],
    question: str,
    session_id: str,
    method: str,
    timeout: int,
) -> Tuple[int, Dict[str, Any]]:
    url = build_url(base_url, endpoint)

    if method.upper() == "POST":
        res = requests.post(
            url,
            headers=headers,
            json={
                "q": question,
                "session_id": session_id,
            },
            timeout=timeout,
        )
    else:
        res = requests.get(
            url,
            headers=headers,
            params={
                "q": question,
                "session_id": session_id,
            },
            timeout=timeout,
        )

    return res.status_code, parse_body(res)


def run_case(
    base_url: str,
    token: str,
    row: Dict[str, str],
    method: str,
    endpoint: str,
    timeout: int,
    delay: float,
) -> Tuple[int, Dict[str, Any], int]:
    headers = build_headers(token, endpoint)

    test_id = row.get("test_id") or row.get("id") or str(int(time.time() * 1000))
    session_id = f"test_{test_id}"

    previous_context = (row.get("previous_context") or "").strip()

    if previous_context:
        send_request(
            base_url=base_url,
            endpoint=endpoint,
            headers=headers,
            question=previous_context,
            session_id=session_id,
            method=method,
            timeout=timeout,
        )

        if delay > 0:
            time.sleep(delay)

    question = (row.get("question") or row.get("query") or "").strip()

    if not question:
        raise ValueError(f"Test case {test_id} không có cột question/query hoặc nội dung đang rỗng.")

    start = time.perf_counter()

    status_code, body = send_request(
        base_url=base_url,
        endpoint=endpoint,
        headers=headers,
        question=question,
        session_id=session_id,
        method=method,
        timeout=timeout,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    if delay > 0:
        time.sleep(delay)

    return status_code, body, latency_ms


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy bộ câu hỏi kiểm thử chatbot RAG/Qdrant."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help=f"File CSV hoặc JSONL chứa câu hỏi test. Mặc định: {DEFAULT_INPUT_FILE}",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"File CSV xuất kết quả. Mặc định: {DEFAULT_OUTPUT_FILE}",
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", DEFAULT_BASE_URL),
        help=f"URL backend FastAPI. Mặc định: {DEFAULT_BASE_URL}",
    )

    parser.add_argument(
        "--token",
        default=os.getenv("FIREBASE_ID_TOKEN", DEFAULT_FIREBASE_ID_TOKEN),
        help="Firebase ID token. Chỉ cần khi --endpoint user.",
    )

    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        choices=["user", "dev"],
        help="user = /user/chat có đăng nhập, dev = /dev/chat local không cần đăng nhập.",
    )

    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=["GET", "POST"],
        help="HTTP method dùng để gọi API chat.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout mỗi câu hỏi, tính bằng giây. Mặc định: {DEFAULT_TIMEOUT}",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Nghỉ giữa các request, tính bằng giây. Mặc định: {DEFAULT_DELAY}",
    )

    args = parser.parse_args()

    if args.endpoint == "user" and not args.token:
        print("[WARN] Đang test /user/chat nhưng chưa có Firebase token.")
        print("[WARN] Có 3 cách xử lý:")
        print("       1. Dán token vào DEFAULT_FIREBASE_ID_TOKEN trong file này.")
        print("       2. Chạy: set FIREBASE_ID_TOKEN=<token>")
        print("       3. Test nhanh bằng: python run_chatbot_tests.py --endpoint dev")
        print("[WARN] Nếu backend yêu cầu đăng nhập, các request /user/chat có thể trả 401.")

    rows = read_rows(args.input)

    if not rows:
        raise ValueError("File input không có dữ liệu.")

    out_fields = list(rows[0].keys()) + [
        "endpoint",
        "http_status",
        "latency_ms",
        "answer",
        "error_detail",
        "auto_need_review",
        "retrieval_top_k_ok",
        "answer_correctness",
        "rejection_correctness",
        "evidence_file",
    ]

    total = len(rows)

    print("=" * 80)
    print("BẮT ĐẦU CHẠY TEST CHATBOT RAG/QDRANT")
    print("=" * 80)
    print(f"Input    : {args.input}")
    print(f"Output   : {args.output}")
    print(f"Base URL : {args.base_url}")
    print(f"Endpoint : {args.endpoint}")
    print(f"Method   : {args.method}")
    print(f"Timeout  : {args.timeout}s")
    print(f"Delay    : {args.delay}s")
    print(f"Cases    : {total}")
    print("=" * 80)

    success_count = 0
    error_count = 0

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            out = dict(row)
            out["endpoint"] = args.endpoint

            try:
                status, body, latency = run_case(
                    base_url=args.base_url,
                    token=args.token,
                    row=row,
                    method=args.method,
                    endpoint=args.endpoint,
                    timeout=args.timeout,
                    delay=args.delay,
                )

                answer = extract_answer(body)
                is_ok = status == 200

                if is_ok:
                    success_count += 1
                else:
                    error_count += 1

                out.update({
                    "http_status": status,
                    "latency_ms": latency,
                    "answer": answer,
                    "error_detail": "" if is_ok else str(body),
                    "auto_need_review": "yes",
                    "retrieval_top_k_ok": "",
                    "answer_correctness": "",
                    "rejection_correctness": "",
                    "evidence_file": "",
                })

            except Exception as e:
                error_count += 1

                out.update({
                    "http_status": "ERROR",
                    "latency_ms": "",
                    "answer": "",
                    "error_detail": str(e),
                    "auto_need_review": "yes",
                    "retrieval_top_k_ok": "",
                    "answer_correctness": "",
                    "rejection_correctness": "",
                    "evidence_file": "",
                })

            writer.writerow(out)

            print(
                f"[{index}/{total}] "
                f"test_id={row.get('test_id', row.get('id', ''))} "
                f"status={out['http_status']} "
                f"latency_ms={out['latency_ms']}"
            )

    print("=" * 80)
    print("HOÀN THÀNH TEST")
    print("=" * 80)
    print(f"Tổng số case : {total}")
    print(f"Thành công   : {success_count}")
    print(f"Lỗi          : {error_count}")
    print(f"File kết quả : {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
