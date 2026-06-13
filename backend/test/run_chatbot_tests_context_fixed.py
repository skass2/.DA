"""
File: run_chatbot_tests_resume_slow.py
Mục đích:
- Chạy bộ câu hỏi kiểm thử chatbot RAG/Qdrant.
- Hỗ trợ CSV và JSONL.
- Hỗ trợ /user/chat có Firebase token và /dev/chat không cần token.
- Truyền procedure_id/procedure_name từ từng dòng test xuống API.
- Có thể chạy tiếp từ file kết quả cũ.
- Có thể chạy lại các test đã HTTP 200 nhưng latency cao.
- Không ghi đè file kết quả cũ, tự tạo bản .no.x nếu file output đã tồn tại.
- Với câu có previous_context, replay toàn bộ ngữ cảnh từ đầu đến cuối trước câu hỏi chính.
- Hỗ trợ nhiều chế độ session để test đúng ngữ cảnh hội thoại:
  + per_case      : mỗi test_id là một phiên riêng như bản cũ.
  + single        : toàn bộ file chạy trong một cuộc trò chuyện.
  + by_procedure  : mỗi thủ tục là một cuộc trò chuyện riêng.
  + by_conversation: ưu tiên cột conversation_id/session_group nếu có, không có thì fallback theo thủ tục.

Cách dùng nhanh:
    py .\run_chatbot_tests_resume_slow.py

Chạy tiếp từ kết quả cũ và chạy lại câu chậm trên 8000ms:
    py .\run_chatbot_tests_resume_slow.py --input test_60_cau.csv --output ket_qua_test_60_cau.csv

Đổi ngưỡng câu chậm:
    py .\run_chatbot_tests_resume_slow.py --rerun-slow-ms 10000

Chỉ resume, không chạy lại câu chậm:
    py .\run_chatbot_tests_resume_slow.py --no-rerun-slow

Chạy lại toàn bộ:
    py .\run_chatbot_tests_resume_slow.py --no-resume --no-rerun-slow

Lưu ý bảo mật:
- Firebase ID Token có hạn dùng ngắn. Nếu gặp 401, lấy token mới từ frontend rồi thay DEFAULT_FIREBASE_ID_TOKEN.
- Không commit file có hard-code token lên GitHub.
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

import requests

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
# Nếu file nằm trong backend/test, vẫn đọc được .env ở backend.
try:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# =========================================================
# CẤU HÌNH MẶC ĐỊNH
# =========================================================
DEFAULT_FIREBASE_ID_TOKEN = os.getenv("FIREBASE_ID_TOKEN", "")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ENDPOINT = "user"       # user = /user/chat, dev = /dev/chat
DEFAULT_METHOD = "GET"
DEFAULT_INPUT_FILE = "test_60_cau_new.csv"
DEFAULT_OUTPUT_FILE = "ket_qua_test_60_cau_new.csv"
DEFAULT_TIMEOUT = 120
DEFAULT_DELAY = 1.0
DEFAULT_RERUN_SLOW_MS = 8000
DEFAULT_SESSION_MODE = "by_procedure"


# =========================================================
# ĐỌC FILE INPUT / OUTPUT CŨ
# =========================================================

def clean_row(row: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Chuẩn hóa row CSV/JSONL để tránh key None làm DictWriter bị lỗi.
    Nếu CSV thừa cột do dấu phẩy chưa quote, phần thừa được gom vào _extra_columns.
    """
    cleaned: Dict[str, Any] = {}
    extra_values: List[str] = []

    for key, value in (row or {}).items():
        if key is None:
            if isinstance(value, list):
                extra_values.extend(str(v) for v in value if v not in (None, ""))
            elif value not in (None, ""):
                extra_values.append(str(value))
            continue

        key_text = str(key)
        if key_text == "_extra_columns" and isinstance(value, list):
            extra_values.extend(str(v) for v in value if v not in (None, ""))
        else:
            cleaned[key_text] = "" if value is None else value

    if extra_values:
        existing = str(cleaned.get("_extra_columns", "") or "").strip()
        joined = " | ".join(extra_values)
        cleaned["_extra_columns"] = f"{existing} | {joined}".strip(" |") if existing else joined

    return cleaned


def read_rows(input_path: str) -> List[Dict[str, Any]]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file input: {input_path}. "
            f"Hãy đặt file test cùng thư mục hoặc truyền --input <duong_dan_file>."
        )

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Lỗi JSONL tại dòng {line_no}: {e}") from e
                rows.append(clean_row(item))
        return rows

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        # restkey giúp tool không chết nếu có dòng CSV thừa cột.
        return [clean_row(row) for row in csv.DictReader(f, restkey="_extra_columns", restval="")]


def get_case_key(row: Dict[str, Any]) -> str:
    """Khóa dùng để resume/rerun. Ưu tiên test_id/id, fallback theo question/query."""
    for key in ("test_id", "id"):
        value = str((row or {}).get(key, "") or "").strip()
        if value:
            return value

    for key in ("question", "query"):
        value = str((row or {}).get(key, "") or "").strip()
        if value:
            return value

    return ""


def is_summary_row(row: Dict[str, Any]) -> bool:
    return any(
        str((row or {}).get(key, "") or "").strip().upper() == "TRUNG_BINH"
        for key in ("test_id", "id")
    )


def parse_latency_ms(row: Dict[str, Any]) -> Optional[int]:
    value = row.get("latency_ms") or row.get("thoi_gian_ms") or ""
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except Exception:
        return None


def is_http_200(row: Dict[str, Any]) -> bool:
    return str((row or {}).get("http_status", "") or "").strip() == "200"


def read_existing_results(result_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not result_path or not result_path.exists():
        return {}

    rows = read_rows(str(result_path))
    results: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        cleaned = clean_row(row)
        if is_summary_row(cleaned):
            continue
        key = get_case_key(cleaned)
        if key:
            # Nếu có nhiều bản ghi cùng test_id, lấy bản cuối cùng.
            results[key] = cleaned

    return results


def next_numbered_output_path(output_path: Path) -> Path:
    """Nếu output đã tồn tại thì tạo ten_file.no.1.csv, ten_file.no.2.csv..."""
    if not output_path.exists():
        return output_path

    stem = output_path.stem
    suffix = output_path.suffix or ".csv"
    parent = output_path.parent

    for index in range(1, 1000):
        candidate = parent / f"{stem}.no.{index}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError("Không tìm được tên file output mới sau 999 phiên bản no.x.")


# =========================================================
# GỌI API
# =========================================================

def build_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if endpoint == "dev":
        return f"{base}/dev/chat"
    return f"{base}/user/chat"


def build_headers(token: str, endpoint: str) -> Dict[str, str]:
    headers = {"accept": "application/json"}
    if endpoint == "user" and token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_body(res: requests.Response) -> Dict[str, Any]:
    try:
        return res.json()
    except Exception:
        return {"raw_text": res.text}


def extract_answer(body: Any) -> str:
    """Lấy phần câu trả lời chính từ nhiều dạng response khác nhau."""
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        return json.dumps(body, ensure_ascii=False)
    if not isinstance(body, dict):
        return str(body)

    direct_keys = [
        "answer", "result", "response", "message", "content", "text",
        "detail", "error", "raw_text",
    ]
    for key in direct_keys:
        value = body.get(key)
        if value not in (None, ""):
            if isinstance(value, (dict, list)):
                return extract_answer(value)
            return str(value)

    for key in ["data", "payload"]:
        value = body.get(key)
        if isinstance(value, (dict, list)):
            extracted = extract_answer(value)
            if extracted:
                return extracted

    return json.dumps(body, ensure_ascii=False)


def send_request(
    base_url: str,
    endpoint: str,
    headers: Dict[str, str],
    question: str,
    session_id: str,
    method: str,
    timeout: int,
    procedure_id: str = "",
    procedure_name: str = "",
) -> Tuple[int, Dict[str, Any]]:
    url = build_url(base_url, endpoint)

    payload = {
        "q": question,
        "session_id": session_id,
        "procedure_id": procedure_id,
        "procedure_name": procedure_name,
    }

    if method.upper() == "POST":
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
    else:
        res = requests.get(url, headers=headers, params=payload, timeout=timeout)

    return res.status_code, parse_body(res)


# =========================================================
# PREVIOUS CONTEXT REPLAY
# =========================================================

def _extract_text_from_context_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        role = str(item.get("role", "") or "").strip().lower()
        # Chỉ replay các lượt người dùng. Lượt assistant là phản hồi cũ, không gửi lại như câu hỏi mới.
        if role in {"assistant", "bot", "model"}:
            return ""
        for key in ("question", "query", "q", "content", "text", "message"):
            value = item.get(key)
            if value not in (None, ""):
                return str(value).strip()
    return str(item).strip()


def split_previous_context(previous_context: str) -> List[str]:
    """
    Replay previous_context từ đầu đến cuối.

    Hỗ trợ:
    - Một câu thường: "Tôi muốn đăng ký..."
    - Nhiều dòng: mỗi dòng là một lượt hỏi.
    - JSON list: ["câu 1", "câu 2"]
    - JSON messages: [{"role":"user","content":"..."}, ...]
    - Separator thủ công: |||, ###, ---CTX---
    """
    text = (previous_context or "").strip()
    if not text:
        return []

    # Thử đọc JSON trước.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            steps = [_extract_text_from_context_item(item) for item in parsed]
            return [s for s in steps if s]
        if isinstance(parsed, dict):
            for key in ("messages", "history", "turns", "contexts", "previous_context"):
                value = parsed.get(key)
                if isinstance(value, list):
                    steps = [_extract_text_from_context_item(item) for item in value]
                    return [s for s in steps if s]
            single = _extract_text_from_context_item(parsed)
            return [single] if single else []
    except Exception:
        pass

    # Split theo separator rõ ràng nếu có.
    separators = ["---CTX---", "|||", "###"]
    for sep in separators:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]

    # Nếu nhiều dòng thì coi mỗi dòng không rỗng là một lượt hỏi.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        cleaned_lines = []
        for line in lines:
            # Bỏ prefix kiểu 1. / - / User: nếu có.
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            line = re.sub(r"^(?:user|người dùng|u|q|question)\s*[:：]\s*", "", line, flags=re.I).strip()
            if line:
                cleaned_lines.append(line)
        return cleaned_lines

    return [text]


def get_procedure_id(row: Dict[str, Any]) -> str:
    return str(
        row.get("procedure_id")
        or row.get("procedure_code")
        or row.get("procedureId")
        or ""
    ).strip()


def get_procedure_name(row: Dict[str, Any]) -> str:
    return str(
        row.get("procedure_name")
        or row.get("procedureName")
        or ""
    ).strip()


def _safe_session_part(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:80] or "unknown"


def get_conversation_id(row: Dict[str, Any]) -> str:
    """
    Lấy mã cuộc trò chuyện nếu file test có khai báo sẵn.
    Các tên cột được hỗ trợ: conversation_id, conversation_group, session_group, group_id, scenario_id.
    """
    for key in ("conversation_id", "conversation_group", "session_group", "group_id", "scenario_id"):
        value = str((row or {}).get(key, "") or "").strip()
        if value:
            return value
    return ""


def make_session_id(
    row: Dict[str, Any],
    suffix: str = "",
    session_mode: str = "per_case",
    global_session_id: str = "",
) -> str:
    """
    Tạo session_id theo chế độ test.

    per_case:
        Mỗi test là một cuộc trò chuyện riêng. Đây là chế độ cũ, phù hợp câu hỏi độc lập.
    single:
        Toàn bộ file là một cuộc trò chuyện. Phù hợp test luồng hội thoại liên tục.
    by_procedure:
        Mỗi procedure_id/procedure_name là một cuộc trò chuyện. Phù hợp bộ test nhiều thủ tục.
    by_conversation:
        Ưu tiên cột conversation_id/session_group; nếu không có thì fallback theo thủ tục.
    """
    mode = (session_mode or "per_case").strip().lower()

    if mode == "single":
        base = global_session_id or "test_single_conversation"

    elif mode == "by_procedure":
        proc_id = get_procedure_id(row)
        proc_name = get_procedure_name(row)
        base = "test_proc_" + _safe_session_part(proc_id or proc_name)

    elif mode == "by_conversation":
        conv_id = get_conversation_id(row)
        if conv_id:
            base = "test_conv_" + _safe_session_part(conv_id)
        else:
            proc_id = get_procedure_id(row)
            proc_name = get_procedure_name(row)
            base = "test_proc_" + _safe_session_part(proc_id or proc_name)

    else:
        test_id = row.get("test_id") or row.get("id") or str(int(time.time() * 1000))
        base = f"test_{test_id}"

    if suffix:
        safe_suffix = _safe_session_part(suffix)
        if safe_suffix:
            return f"{base}_{safe_suffix}"

    return base


def run_case(
    base_url: str,
    token: str,
    row: Dict[str, Any],
    method: str,
    endpoint: str,
    timeout: int,
    delay: float,
    session_suffix: str = "",
    verbose_context: bool = True,
    session_mode: str = "per_case",
    global_session_id: str = "",
    sent_context_cache: Optional[set] = None,
) -> Tuple[int, Dict[str, Any], int, Dict[str, Any]]:
    headers = build_headers(token, endpoint)
    session_id = make_session_id(
        row,
        suffix=session_suffix,
        session_mode=session_mode,
        global_session_id=global_session_id,
    )
    procedure_id = get_procedure_id(row)
    procedure_name = get_procedure_name(row)

    previous_context = str(row.get("previous_context") or "").strip()
    context_steps = split_previous_context(previous_context)

    context_statuses: List[str] = []
    context_latencies: List[int] = []

    for ctx_index, ctx_question in enumerate(context_steps, start=1):
        context_cache_key = (session_id, ctx_question)
        if sent_context_cache is not None and context_cache_key in sent_context_cache:
            context_statuses.append("SKIP_DUP_CONTEXT")
            context_latencies.append(0)
            if verbose_context:
                print(
                    f"    [context {ctx_index}/{len(context_steps)}] "
                    f"status=SKIP_DUP_CONTEXT latency_ms=0 q={ctx_question[:80]}"
                )
            continue

        ctx_start = time.perf_counter()
        try:
            ctx_status, _ctx_body = send_request(
                base_url=base_url,
                endpoint=endpoint,
                headers=headers,
                question=ctx_question,
                session_id=session_id,
                method=method,
                timeout=timeout,
                procedure_id=procedure_id,
                procedure_name=procedure_name,
            )
            ctx_latency = int((time.perf_counter() - ctx_start) * 1000)
            context_statuses.append(str(ctx_status))
            context_latencies.append(ctx_latency)
            if sent_context_cache is not None and str(ctx_status) == "200":
                sent_context_cache.add(context_cache_key)
            if verbose_context:
                print(
                    f"    [context {ctx_index}/{len(context_steps)}] "
                    f"status={ctx_status} latency_ms={ctx_latency} q={ctx_question[:80]}"
                )
        except Exception as e:
            ctx_latency = int((time.perf_counter() - ctx_start) * 1000)
            context_statuses.append(f"ERROR:{e}")
            context_latencies.append(ctx_latency)
            if verbose_context:
                print(
                    f"    [context {ctx_index}/{len(context_steps)}] "
                    f"status=ERROR latency_ms={ctx_latency} error={e}"
                )

        if delay > 0:
            time.sleep(delay)

    question = str(row.get("question") or row.get("query") or "").strip()
    if not question:
        test_id = row.get("test_id") or row.get("id") or ""
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
        procedure_id=procedure_id,
        procedure_name=procedure_name,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    if delay > 0:
        time.sleep(delay)

    context_info = {
        "session_id_used": session_id,
        "context_replay_count": len(context_steps),
        "context_replay_statuses": " | ".join(context_statuses),
        "context_replay_latencies_ms": " | ".join(str(x) for x in context_latencies),
        "context_replay_total_ms": sum(context_latencies) if context_latencies else "",
    }
    return status_code, body, latency_ms, context_info


# =========================================================
# XỬ LÝ OUTPUT
# =========================================================

def unique_fields(fields: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for field in fields:
        if field and field not in seen:
            seen.add(field)
            result.append(field)
    return result


def make_summary_row(
    out_fields: List[str],
    rows: List[Dict[str, Any]],
    total: int,
    success_count: int,
    error_count: int,
    skipped_count: int,
    rerun_slow_count: int,
    success_latencies: List[int],
    all_latencies: List[int],
) -> Dict[str, Any]:
    summary = {field: "" for field in out_fields}
    avg_source = success_latencies if success_latencies else all_latencies
    avg_latency_ms = round(sum(avg_source) / len(avg_source), 2) if avg_source else ""
    avg_latency_s = round(float(avg_latency_ms) / 1000, 3) if avg_latency_ms != "" else ""

    first_input_field = next(iter(rows[0].keys()), "test_id") if rows else "test_id"
    if first_input_field in summary:
        summary[first_input_field] = "TRUNG_BINH"
    if "test_id" in summary:
        summary["test_id"] = "TRUNG_BINH"
    if "question" in summary:
        summary["question"] = "Dòng tổng kết thời gian chạy test"

    summary.update({
        "endpoint": "SUMMARY",
        "http_status": f"OK={success_count}/{total}; ERROR={error_count}; SKIP={skipped_count}; RERUN_SLOW={rerun_slow_count}",
        "ket_qua": "Thời gian trung bình tính trên các case HTTP 200 được ghi trong file output.",
        "answer": "SUMMARY",
        "latency_ms": avg_latency_ms,
        "thoi_gian_ms": avg_latency_ms,
        "thoi_gian_giay": avg_latency_s,
        "avg_latency_ms": avg_latency_ms,
    })
    return summary


def decide_run_mode(
    row: Dict[str, Any],
    old_result: Optional[Dict[str, Any]],
    no_resume: bool,
    rerun_slow: bool,
    slow_ms: int,
) -> Tuple[str, str, Optional[int]]:
    """
    Trả về: run_mode, reason, previous_latency_ms.
    - RUN_NEW: chưa có kết quả cũ.
    - RERUN_ERROR: kết quả cũ không phải HTTP 200.
    - RERUN_SLOW: kết quả cũ HTTP 200 nhưng latency >= slow_ms.
    - SKIP_EXISTING_200: đã có kết quả ổn và không chậm.
    - RUN_FORCED: --no-resume.
    """
    if no_resume:
        return "RUN_FORCED", "no_resume", None

    if not old_result:
        return "RUN_NEW", "missing_result", None

    old_latency = parse_latency_ms(old_result)

    if not is_http_200(old_result):
        return "RERUN_ERROR", f"old_http_status={old_result.get('http_status', '')}", old_latency

    if rerun_slow and old_latency is not None and old_latency >= slow_ms:
        return "RERUN_SLOW", f"old_latency_ms={old_latency} >= {slow_ms}", old_latency

    return "SKIP_EXISTING_200", "old_http_200_and_not_slow", old_latency


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy bộ câu hỏi kiểm thử chatbot RAG/Qdrant.")

    parser.add_argument("--input", default=DEFAULT_INPUT_FILE,
        help=f"File CSV hoặc JSONL chứa câu hỏi test. Mặc định: {DEFAULT_INPUT_FILE}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE,
        help=f"File CSV xuất kết quả. Mặc định: {DEFAULT_OUTPUT_FILE}")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL),
        help=f"URL backend FastAPI. Mặc định: {DEFAULT_BASE_URL}")
    parser.add_argument("--token", default=os.getenv("FIREBASE_ID_TOKEN", DEFAULT_FIREBASE_ID_TOKEN),
        help="Firebase ID token. Chỉ cần khi --endpoint user.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, choices=["user", "dev"],
        help="user = /user/chat có đăng nhập, dev = /dev/chat local không cần đăng nhập.")
    parser.add_argument("--method", default=DEFAULT_METHOD, choices=["GET", "POST"],
        help="HTTP method dùng để gọi API chat.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Timeout mỗi câu hỏi, tính bằng giây. Mặc định: {DEFAULT_TIMEOUT}")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
        help=f"Nghỉ giữa các request, tính bằng giây. Mặc định: {DEFAULT_DELAY}")
    parser.add_argument("--no-summary-row", action="store_true",
        help="Không ghi dòng tổng kết trung bình thời gian ở cuối file CSV.")

    parser.add_argument("--no-resume", action="store_true",
        help="Không bỏ qua test đã chạy, ép chạy lại toàn bộ.")
    parser.add_argument("--resume-from", default="",
        help="File kết quả cũ dùng để resume/rerun slow. Nếu bỏ trống, dùng --output hiện có.")
    parser.add_argument("--overwrite", action="store_true",
        help="Cho phép ghi đè file --output. Mặc định nếu file đã có thì tạo bản .no.x.")

    parser.add_argument("--rerun-slow-ms", type=int, default=DEFAULT_RERUN_SLOW_MS,
        help=f"Chạy lại các test có latency >= ngưỡng này trong file kết quả cũ. Mặc định: {DEFAULT_RERUN_SLOW_MS}ms.")
    parser.add_argument("--no-rerun-slow", action="store_true",
        help="Tắt chế độ chạy lại test chậm, chỉ resume test thiếu/lỗi.")
    parser.add_argument("--fresh-session-for-rerun", action="store_true", default=True,
        help="Khi chạy lại test chậm/lỗi, dùng session_id mới để tránh dính lịch sử cũ. Mặc định bật.")
    parser.add_argument("--keep-session-for-rerun", dest="fresh_session_for_rerun", action="store_false",
        help="Khi chạy lại test chậm/lỗi, vẫn dùng session_id cũ.")

    parser.add_argument("--session-mode", default=DEFAULT_SESSION_MODE,
        choices=["per_case", "single", "by_procedure", "by_conversation"],
        help=(
            "Cách gom các câu test vào phiên hội thoại. "
            "per_case=mỗi câu một session như bản cũ; "
            "single=toàn bộ file một session; "
            "by_procedure=mỗi thủ tục một session; "
            "by_conversation=theo cột conversation_id/session_group nếu có, không có thì theo thủ tục. "
            f"Mặc định: {DEFAULT_SESSION_MODE}"
        ))
    parser.add_argument("--global-session-id", default="",
        help="Session id dùng khi --session-mode single. Nếu bỏ trống, tool tự tạo session mới theo thời điểm chạy.")
    parser.add_argument("--no-dedupe-context", action="store_true",
        help="Không bỏ qua previous_context trùng lặp trong cùng session. Mặc định tool sẽ tránh replay lặp cùng một context.")

    args = parser.parse_args()

    if args.endpoint == "user" and not args.token:
        print("[WARN] Đang test /user/chat nhưng chưa có Firebase token.")
        print("[WARN] Có 3 cách xử lý:")
        print("       1. Truyền --token <firebase_id_token>")
        print("       2. PowerShell: $env:FIREBASE_ID_TOKEN='<firebase_id_token>'")
        print("       3. Test nhanh bằng: py .\\run_chatbot_tests_resume_slow.py --endpoint dev")
        print("[WARN] Nếu backend yêu cầu đăng nhập, các request /user/chat có thể trả 401.")

    rows = read_rows(args.input)
    if not rows:
        raise ValueError("File input không có dữ liệu.")

    requested_output_path = Path(args.output)
    resume_source_path: Optional[Path] = None

    if not args.no_resume:
        if args.resume_from:
            resume_source_path = Path(args.resume_from)
        elif requested_output_path.exists() and not args.overwrite:
            resume_source_path = requested_output_path

    existing_results = read_existing_results(resume_source_path)
    output_path = requested_output_path if args.overwrite else next_numbered_output_path(requested_output_path)

    existing_field_names: List[str] = []
    for old_row in existing_results.values():
        existing_field_names.extend(list(old_row.keys()))

    out_fields = unique_fields(list(rows[0].keys()) + existing_field_names + [
        "_extra_columns",
        "endpoint",
        "run_mode",
        "rerun_reason",
        "previous_latency_ms",
        "session_id_used",
        "context_replay_count",
        "context_replay_statuses",
        "context_replay_latencies_ms",
        "context_replay_total_ms",
        "http_status",
        "ket_qua",
        "thoi_gian_ms",
        "thoi_gian_giay",
        "answer",
        "latency_ms",
        "error_detail",
        "auto_need_review",
        "retrieval_top_k_ok",
        "answer_correctness",
        "rejection_correctness",
        "evidence_file",
        "avg_latency_ms",
    ])

    total = len(rows)
    rerun_slow_enabled = not args.no_rerun_slow and args.rerun_slow_ms > 0

    # Tạo session dùng chung trước khi in log.
    # Bản trước in global_session_id trước khi gán nên bị UnboundLocalError khi --session-mode single.
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    global_session_id = args.global_session_id or f"test_single_{run_stamp}"
    sent_context_cache = None if args.no_dedupe_context else set()

    print("=" * 80)
    print("BẮT ĐẦU CHẠY TEST CHATBOT RAG/QDRANT")
    print("=" * 80)
    print(f"Input       : {args.input}")
    print(f"Output      : {output_path}")
    if resume_source_path:
        print(f"Resume from : {resume_source_path} | existing rows: {len(existing_results)}")
    print(f"Base URL    : {args.base_url}")
    print(f"Endpoint    : {args.endpoint}")
    print(f"Method      : {args.method}")
    print(f"Timeout     : {args.timeout}s")
    print(f"Delay       : {args.delay}s")
    print(f"Cases       : {total}")
    print(f"Rerun slow  : {rerun_slow_enabled} | threshold={args.rerun_slow_ms}ms")
    print(f"Session mode: {args.session_mode}")
    if args.session_mode == "single":
        print(f"Global SID  : {global_session_id}")
    print("=" * 80)

    success_count = 0
    error_count = 0
    skipped_count = 0
    rerun_slow_count = 0
    all_latencies: List[int] = []
    success_latencies: List[int] = []

    if args.session_mode != "per_case" and not args.no_resume:
        print("[WARN] Bố đang dùng session theo hội thoại nhưng vẫn bật resume/skip.")
        print("[WARN] Nếu cần chấm công bằng theo ngữ cảnh, nên chạy full: --no-resume --no-rerun-slow")

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()

        for index, raw_row in enumerate(rows, start=1):
            row = clean_row(raw_row)
            out = dict(row)
            out["endpoint"] = args.endpoint

            case_key = get_case_key(row)
            old_result = existing_results.get(case_key) if case_key else None
            run_mode, reason, previous_latency = decide_run_mode(
                row=row,
                old_result=old_result,
                no_resume=args.no_resume,
                rerun_slow=rerun_slow_enabled,
                slow_ms=args.rerun_slow_ms,
            )

            out["run_mode"] = run_mode
            out["rerun_reason"] = reason
            out["previous_latency_ms"] = "" if previous_latency is None else previous_latency

            if run_mode == "SKIP_EXISTING_200" and old_result:
                out.update(old_result)
                out["run_mode"] = run_mode
                out["rerun_reason"] = reason
                out["previous_latency_ms"] = "" if previous_latency is None else previous_latency
                latency = parse_latency_ms(out)
                if latency is not None:
                    all_latencies.append(latency)
                    success_latencies.append(latency)
                success_count += 1
                skipped_count += 1
                writer.writerow(out)
                print(
                    f"[{index}/{total}] test_id={case_key} "
                    f"status=SKIP_EXISTING_200 latency_ms={out.get('latency_ms', out.get('thoi_gian_ms', ''))}"
                )
                continue

            if run_mode == "RERUN_SLOW":
                rerun_slow_count += 1

            session_suffix = ""
            if args.fresh_session_for_rerun and run_mode in {"RERUN_SLOW", "RERUN_ERROR"}:
                session_suffix = f"{run_mode.lower()}_{run_stamp}"

            try:
                status, body, latency, context_info = run_case(
                    base_url=args.base_url,
                    token=args.token,
                    row=row,
                    method=args.method,
                    endpoint=args.endpoint,
                    timeout=args.timeout,
                    delay=args.delay,
                    session_suffix=session_suffix,
                    verbose_context=True,
                    session_mode=args.session_mode,
                    global_session_id=global_session_id,
                    sent_context_cache=sent_context_cache,
                )

                answer = extract_answer(body)
                is_ok = status == 200
                all_latencies.append(latency)

                if is_ok:
                    success_count += 1
                    success_latencies.append(latency)
                else:
                    error_count += 1

                ket_qua = answer if is_ok else extract_answer(body)
                out.update(context_info)
                out.update({
                    "http_status": status,
                    "ket_qua": ket_qua,
                    "thoi_gian_ms": latency,
                    "thoi_gian_giay": round(latency / 1000, 3),
                    "answer": answer,
                    "latency_ms": latency,
                    "error_detail": "" if is_ok else json.dumps(body, ensure_ascii=False),
                    "auto_need_review": "yes",
                    "retrieval_top_k_ok": "",
                    "answer_correctness": "",
                    "rejection_correctness": "",
                    "evidence_file": "",
                    "avg_latency_ms": "",
                })

            except Exception as e:
                error_count += 1
                error_message = str(e)
                out.update({
                    "http_status": "ERROR",
                    "ket_qua": error_message,
                    "thoi_gian_ms": "",
                    "thoi_gian_giay": "",
                    "answer": "",
                    "latency_ms": "",
                    "error_detail": error_message,
                    "auto_need_review": "yes",
                    "retrieval_top_k_ok": "",
                    "answer_correctness": "",
                    "rejection_correctness": "",
                    "evidence_file": "",
                    "avg_latency_ms": "",
                })

            writer.writerow(out)
            print(
                f"[{index}/{total}] test_id={case_key} "
                f"mode={run_mode} status={out['http_status']} latency_ms={out['latency_ms']}"
            )

        if not args.no_summary_row:
            summary_row = make_summary_row(
                out_fields=out_fields,
                rows=rows,
                total=total,
                success_count=success_count,
                error_count=error_count,
                skipped_count=skipped_count,
                rerun_slow_count=rerun_slow_count,
                success_latencies=success_latencies,
                all_latencies=all_latencies,
            )
            writer.writerow(summary_row)

    avg_source = success_latencies if success_latencies else all_latencies
    avg_latency_ms = round(sum(avg_source) / len(avg_source), 2) if avg_source else "N/A"

    print("=" * 80)
    print("HOÀN THÀNH TEST")
    print("=" * 80)
    print(f"Tổng số case       : {total}")
    print(f"Thành công         : {success_count}")
    print(f"Lỗi                : {error_count}")
    print(f"Skip đã nhanh      : {skipped_count}")
    print(f"Rerun câu chậm     : {rerun_slow_count}")
    print(f"Trung bình latency : {avg_latency_ms} ms")
    print(f"File kết quả       : {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
