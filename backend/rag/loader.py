import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DATA_FILES = [
    Path("data/procedures_dvc_new.json"),
    Path("data/procedures_dvc_new_backup.json"),
    Path("data/procedures.json"),
]

INVALID_NAME_VALUES = {
    "",
    "--",
    "không",
    "không có",
    "không có thông tin",
    "không quy định",
    "chưa phân loại",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def short_code(code: str) -> str:
    code = clean_text(code)
    parts = code.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return code


def _as_list(data: Any) -> List[Dict[str, Any]]:
    """
    Hỗ trợ cả 2 dạng:
    - [ {...}, {...} ]
    - {"procedures": [ ... ]} hoặc {"results": [ ... ]}
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ["procedures", "results", "data", "items"]:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    return []


def get_content(item: Dict[str, Any]) -> Dict[str, Any]:
    content = item.get("content", {})
    return content if isinstance(content, dict) else {}


def get_procedure_name(item: Dict[str, Any]) -> str:
    content = get_content(item)
    return clean_text(item.get("name") or content.get("Tên thủ tục") or "")


def get_procedure_id(item: Dict[str, Any]) -> str:
    content = get_content(item)
    return clean_text(item.get("id") or content.get("Mã thủ tục") or item.get("search_code") or "")


def is_invalid_name(value: Any) -> bool:
    name = clean_text(value).lower()
    return name in INVALID_NAME_VALUES


def is_error_record(item: Dict[str, Any]) -> bool:
    """
    Bản ghi lỗi do crawler sinh ra, ví dụ:
    {
      "id": "2.000622.000.00.00.H35",
      "name": "",
      "error": "Không tìm thấy kết quả cho mã 2.000622"
    }
    """
    if not isinstance(item, dict):
        return True

    if clean_text(item.get("error")):
        return True

    # Bản ghi crawl hỏng thường không có content và name rỗng.
    if not get_content(item) and not get_procedure_name(item):
        return True

    return False


def is_valid_procedure(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False

    if is_error_record(item):
        return False

    name = get_procedure_name(item)
    if not name or is_invalid_name(name):
        return False

    content = get_content(item)
    if not isinstance(content, dict) or not content:
        return False

    return True


def normalize_procedure(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn hóa nhẹ để các API/chunker không bị trắng tên, thiếu mã rút gọn.
    Không sửa nội dung pháp lý, chỉ bổ sung khóa kỹ thuật còn thiếu.
    """
    normalized = dict(item)
    content = dict(get_content(item))

    name = get_procedure_name(item)
    p_id = get_procedure_id(item)
    search_code = clean_text(item.get("search_code") or short_code(content.get("Mã thủ tục") or p_id))

    normalized["id"] = p_id
    normalized["name"] = name
    normalized["search_code"] = search_code

    if name and not clean_text(content.get("Tên thủ tục")):
        content["Tên thủ tục"] = name
    if search_code and not clean_text(content.get("Mã thủ tục")):
        content["Mã thủ tục"] = search_code
    if clean_text(item.get("detail_url")) and not clean_text(content.get("source_url")):
        content["source_url"] = clean_text(item.get("detail_url"))

    normalized["content"] = content
    return normalized


def _choose_data_path(path: Optional[str] = None) -> Path:
    if path:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {candidate}")

    for candidate in DEFAULT_DATA_FILES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Không tìm thấy dữ liệu thủ tục. Cần có data/procedures_dvc_new.json, "
        "data/procedures_dvc_new_backup.json hoặc data/procedures.json."
    )


def load_raw_data(path: Optional[str] = None) -> List[Dict[str, Any]]:
    data_path = _choose_data_path(path)
    with open(data_path, "r", encoding="utf-8") as f:
        return _as_list(json.load(f))


def filter_valid_procedures(data: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    valid: List[Dict[str, Any]] = []
    stats = {
        "total": 0,
        "valid": 0,
        "skipped": 0,
        "error_records": 0,
        "empty_name": 0,
        "empty_content": 0,
    }

    for item in data:
        stats["total"] += 1

        if not isinstance(item, dict):
            stats["skipped"] += 1
            continue

        if clean_text(item.get("error")):
            stats["error_records"] += 1
            stats["skipped"] += 1
            continue

        if not get_content(item):
            stats["empty_content"] += 1
            stats["skipped"] += 1
            continue

        name = get_procedure_name(item)
        if not name or is_invalid_name(name):
            stats["empty_name"] += 1
            stats["skipped"] += 1
            continue

        valid.append(normalize_procedure(item))
        stats["valid"] += 1

    return valid, stats


def load_data(path: Optional[str] = None, include_invalid: bool = False) -> List[Dict[str, Any]]:
    """
    Mặc định chỉ trả về thủ tục hợp lệ.

    Nhờ vậy các luồng sau tự né bản ghi lỗi:
    - admin reload vector DB;
    - chunker;
    - HomePage/search API;
    - ProcedureDetail;
    - pipeline đọc danh sách thủ tục.

    Khi thật sự cần xem file gốc, truyền include_invalid=True.
    """
    raw = load_raw_data(path)
    if include_invalid:
        return raw

    valid, stats = filter_valid_procedures(raw)
    print(
        "[LOADER] Loaded "
        f"{stats['valid']}/{stats['total']} valid procedures "
        f"(skipped={stats['skipped']}, errors={stats['error_records']}, "
        f"empty_name={stats['empty_name']}, empty_content={stats['empty_content']})."
    )
    return valid


def get_data_health_report(path: Optional[str] = None) -> Dict[str, Any]:
    raw = load_raw_data(path)
    valid, stats = filter_valid_procedures(raw)

    samples = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if is_valid_procedure(item):
            continue
        samples.append({
            "id": clean_text(item.get("id")),
            "search_code": clean_text(item.get("search_code")),
            "name": clean_text(item.get("name")),
            "error": clean_text(item.get("error")),
        })
        if len(samples) >= 20:
            break

    return {
        **stats,
        "valid_ratio": round(stats["valid"] / stats["total"], 4) if stats["total"] else 0,
        "sample_invalid_records": samples,
    }
