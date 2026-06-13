"""
Crawler mới cho Cổng DVC Quốc gia:
- Vào https://dichvucong.gov.vn/thu-tuc-hanh-chinh
- Tìm kiếm theo mã rút gọn, ví dụ 2.002307.000.00.00.H35 -> 2.002307
- Mở đúng trang chi tiết thủ tục
- Cào đầy đủ các trường hiện trên trang mới
- Tải file mẫu trong mục Thành phần hồ sơ vào thư mục file_Mau
- Gắn móc nối file mẫu vào JSON để sau này upload server vẫn nối được với hồ sơ

Cài đặt:
    pip install playwright beautifulsoup4
    python -m playwright install chromium

Chạy:
    python getData_dvc_new_with_forms.py --input all_laichau_codes.json --out data/procedures_dvc_new.json

Chạy thử một vài mã:
    python getData_dvc_new_with_forms.py --limit 5

Test parser bằng HTML đã lưu sẵn:
    python getData_dvc_new_with_forms.py --local-html "nội dung thủ tục.html" --out data/test_one.json
"""

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag


BASE_URL = "https://dichvucong.gov.vn/thu-tuc-hanh-chinh"
DEFAULT_INPUT = "all_laichau_codes.json"
DEFAULT_OUT = "data/procedures_dvc_new.json"
DEFAULT_FORM_DIR = "data/file_Mau"


TOP_LEVEL_GRID_FIELDS = {
    "Tên thủ tục": "Tên thủ tục",
    "Mã thủ tục": "Mã thủ tục",
    "Số quyết định": "Số quyết định",
    "Cấp thực hiện": "Cấp thực hiện",
    "Loại thủ tục": "Loại thủ tục",
    "Lĩnh vực": "Lĩnh vực",
    "Đối tượng thực hiện": "Đối tượng thực hiện",
    "Cơ quan có thẩm quyền": "Cơ quan có thẩm quyền",
    "Địa chỉ tiếp nhận HS": "Địa chỉ tiếp nhận HS",
    "Cơ quan được ủy quyền": "Cơ quan được ủy quyền",
    "Cơ quan phối hợp": "Cơ quan phối hợp",
}


def norm_text(value: str) -> str:
    """Chuẩn hóa text để so sánh tiêu đề/nhãn."""
    if value is None:
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    return value


def get_text(el: Optional[Tag]) -> str:
    if not el:
        return ""
    return norm_text(el.get_text(separator="\n", strip=True))


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def slugify(text: str, max_len: int = 90) -> str:
    """Tạo tên file an toàn trên Windows/Linux."""
    text = strip_accents(text or "")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "thu-tuc")


def short_procedure_code(full_code: str) -> str:
    """
    Rút gọn mã theo yêu cầu:
    2.002307.000.00.00.H35 -> 2.002307
    1.014028.H35 -> 1.014028
    """
    parts = (full_code or "").split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return full_code or ""


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_section(soup: BeautifulSoup, title: str) -> Optional[Tag]:
    """Tìm container của một mục theo h4, ví dụ 'Thành phần hồ sơ'."""
    target = norm_text(title).lower()
    h = soup.find(
        lambda tag: isinstance(tag, Tag)
        and tag.name in {"h3", "h4", "h5"}
        and norm_text(tag.get_text(" ", strip=True)).lower() == target
    )
    return h.parent if h else None


def remove_first_heading_text(container: Tag) -> str:
    """Lấy text của section nhưng bỏ chính tiêu đề section."""
    clone = BeautifulSoup(str(container), "html.parser")
    first_heading = clone.find(["h3", "h4", "h5"])
    if first_heading:
        first_heading.decompose()
    return get_text(clone)


def parse_top_grid(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Parse phần lưới đầu trang.
    Trang mới render dạng nhiều dòng, mỗi dòng có 2 div con: nhãn và giá trị.
    """
    data: Dict[str, str] = {}

    for row in soup.find_all("div"):
        children = [c for c in row.find_all("div", recursive=False)]
        if len(children) != 2:
            continue

        key = norm_text(children[0].get_text(" ", strip=True))
        value = norm_text(children[1].get_text("\n", strip=True))

        if key in TOP_LEVEL_GRID_FIELDS and value:
            data[TOP_LEVEL_GRID_FIELDS[key]] = value

    return data


def parse_simple_section_text(soup: BeautifulSoup, title: str) -> str:
    section = find_section(soup, title)
    if not section:
        return ""
    return remove_first_heading_text(section)


def table_rows(table: Tag) -> List[List[Tag]]:
    rows = []
    for tr in table.select("tbody tr"):
        cols = tr.find_all("td", recursive=False)
        if cols:
            rows.append(cols)
    return rows


def parse_fee_value(fee_text: str, fee_list: List[Dict[str, str]], lephi_list: List[Dict[str, str]]) -> List[str]:
    """
    Cột trên site mới là 'Phí, lệ phí', có thể là 'Miễn phí', 'Không quy định',
    hoặc đoạn dài. Để giữ dữ liệu sạch, vẫn tạo id móc nối.
    """
    fee_text = norm_text(fee_text)
    if not fee_text:
        return []

    lower = fee_text.lower()
    if "lệ phí" in lower:
        obj = {"id": f"LP{len(lephi_list) + 1}", "text": fee_text}
        lephi_list.append(obj)
    else:
        obj = {"id": f"P{len(fee_list) + 1}", "text": fee_text}
        fee_list.append(obj)
    return [obj["id"]]


def parse_methods(soup: BeautifulSoup) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, str]]]:
    section = find_section(soup, "Cách Thức Thực Hiện")
    methods: List[Dict[str, Any]] = []
    fees: List[Dict[str, str]] = []
    lephis: List[Dict[str, str]] = []

    if not section:
        return methods, fees, lephis

    table = section.find("table")
    if not table:
        return methods, fees, lephis

    for cols in table_rows(table):
        if len(cols) < 4:
            continue

        fee_ids = parse_fee_value(get_text(cols[2]), fees, lephis)
        methods.append({
            "Hình thức": get_text(cols[0]),
            "Thời hạn": get_text(cols[1]),
            "Phí, lệ phí": get_text(cols[2]),
            "Mô tả": get_text(cols[3]),
            "Liên kết phí": fee_ids,
        })

    return methods, fees, lephis


def split_template_names(text: str) -> List[str]:
    """
    Tách tên file mẫu trong ô 'Mẫu đơn, tờ khai'.
    Hỗ trợ nhiều file trong một ô: .doc, .docx, .pdf, .xls, .xlsx, .zip...
    """
    text = text or ""
    candidates = re.split(r"[\n;,]+", text)
    results = []
    for item in candidates:
        item = norm_text(item)
        if not item:
            continue
        if re.search(r"\.(docx?|xlsx?|pdf|zip|rar|odt|rtf)$", item, flags=re.I):
            results.append(item)
    return results


def parse_dossier_without_downloads(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    section = find_section(soup, "Thành phần hồ sơ")
    items: List[Dict[str, Any]] = []

    if not section:
        return items

    # Có thể chia theo nhóm hồ sơ, mỗi nhóm có h5, bên trong là một table.
    groups = section.select("div.border.border-gray-200")
    if not groups:
        groups = [section]

    for group in groups:
        group_name_el = group.find("h5")
        group_name = get_text(group_name_el) or "--"
        table = group.find("table")
        if not table:
            continue

        for row_index, cols in enumerate(table_rows(table), start=1):
            if len(cols) < 3:
                continue

            template_text = get_text(cols[1])
            templates = split_template_names(template_text)
            items.append({
                "Nhóm hồ sơ": group_name,
                "Tên giấy tờ": get_text(cols[0]),
                "Biểu mẫu": template_text,
                "Số lượng": get_text(cols[2]),
                "Mẫu đính kèm": [
                    {
                        "template_id": "",
                        "order": 0,
                        "original_name": name,
                        "stored_name": "",
                        "relative_path": "",
                        "download_status": "found_text_only",
                        "source_url": "",
                    }
                    for name in templates
                ],
            })

    return items


def parse_legal_basis(soup: BeautifulSoup) -> List[Dict[str, str]]:
    section = find_section(soup, "Căn cứ pháp lý")
    legal: List[Dict[str, str]] = []
    if not section:
        return legal

    table = section.find("table")
    if not table:
        return legal

    for cols in table_rows(table):
        if len(cols) >= 2:
            legal.append({
                "Tên văn bản": get_text(cols[0]),
                "Số hiệu": get_text(cols[1]),
                "Ngày ban hành": "",
                "Cơ quan ban hành": "",
            })
    return legal


def parse_results(soup: BeautifulSoup) -> List[Dict[str, str]]:
    section = find_section(soup, "Kết quả xử lý")
    if not section:
        return []

    results = []
    # Mỗi kết quả thường có tên và dòng Mã: ...
    for block in section.select("div.pl-3"):
        text_lines = [x.strip() for x in block.get_text("\n", strip=True).split("\n") if x.strip()]
        if not text_lines:
            continue
        code = ""
        name = text_lines[0]
        for line in text_lines[1:]:
            if line.lower().startswith("mã:"):
                code = line.split(":", 1)[-1].strip()
        results.append({"Tên kết quả": name, "Mã kết quả": code})

    if not results:
        raw = remove_first_heading_text(section)
        if raw:
            results.append({"Tên kết quả": raw, "Mã kết quả": ""})

    return results


def build_content_from_html(html: str, page_url: str = "") -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    grid = parse_top_grid(soup)
    methods, fees, lephis = parse_methods(soup)
    dossier = parse_dossier_without_downloads(soup)
    legal = parse_legal_basis(soup)
    results = parse_results(soup)

    result_text = ", ".join([r["Tên kết quả"] for r in results if r.get("Tên kết quả")])

    content = {
        "Tên thủ tục": grid.get("Tên thủ tục", ""),
        "Mã thủ tục": grid.get("Mã thủ tục", ""),
        "Số quyết định": grid.get("Số quyết định", ""),
        "Cấp thực hiện": grid.get("Cấp thực hiện", ""),
        "Loại thủ tục": grid.get("Loại thủ tục", ""),
        "Lĩnh vực": grid.get("Lĩnh vực", ""),
        "Đối tượng thực hiện": grid.get("Đối tượng thực hiện", ""),
        "Cơ quan thực hiện": grid.get("Cơ quan có thẩm quyền", "") or grid.get("Địa chỉ tiếp nhận HS", ""),
        "Cơ quan có thẩm quyền": grid.get("Cơ quan có thẩm quyền", ""),
        "Địa chỉ tiếp nhận HS": grid.get("Địa chỉ tiếp nhận HS", ""),
        "Cơ quan được ủy quyền": grid.get("Cơ quan được ủy quyền", ""),
        "Cơ quan phối hợp": grid.get("Cơ quan phối hợp", ""),
        "Thủ tục hành chính liên quan": parse_simple_section_text(soup, "Thủ tục hành chính liên quan"),
        "Trình tự thực hiện": parse_simple_section_text(soup, "Trình Tự Thực Hiện"),
        "Cách thức thực hiện": methods,
        "Phí": fees,
        "Lệ phí": lephis,
        "Thành phần hồ sơ": dossier,
        "Căn cứ pháp lý": legal,
        "Yêu cầu điều kiện": parse_simple_section_text(soup, "Yêu cầu, điều kiện thực hiện"),
        "Kết quả thực hiện": result_text,
        "Kết quả xử lý": results,
        "Từ khóa": parse_simple_section_text(soup, "Từ khóa"),
        "Mô tả": parse_simple_section_text(soup, "Mô tả"),
        "source_url": page_url,
    }

    content["File mẫu"] = collect_all_forms(dossier)
    return content


def make_form_filename(proc_code_short: str, procedure_name: str, index: int, suggested_name: str = "") -> str:
    ext = ""
    if suggested_name:
        suffix = Path(suggested_name).suffix
        if suffix:
            ext = suffix.lower()

    if not ext:
        ext = ".bin"

    return f"{proc_code_short}-{slugify(procedure_name)}-m{index}{ext}"


def update_dossier_with_download_result(
    dossier: List[Dict[str, Any]],
    row_index: int,
    template_index_in_row: int,
    template_meta: Dict[str, Any],
) -> None:
    if row_index >= len(dossier):
        return

    row = dossier[row_index]
    templates = row.setdefault("Mẫu đính kèm", [])

    while len(templates) <= template_index_in_row:
        templates.append({
            "template_id": "",
            "order": 0,
            "original_name": "",
            "stored_name": "",
            "relative_path": "",
            "download_status": "not_found",
            "source_url": "",
        })

    templates[template_index_in_row].update(template_meta)


def collect_all_forms(dossier: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    forms = []
    for row in dossier:
        for form in row.get("Mẫu đính kèm", []):
            if form.get("original_name") or form.get("stored_name"):
                forms.append(form)
    return forms


def save_progress(out_path: str | Path, data: List[Dict[str, Any]]) -> None:
    write_json(out_path, data)
    backup = str(out_path).replace(".json", "_backup.json")
    write_json(backup, data)


def parse_local_html(local_html: str | Path, out_path: str | Path) -> None:
    html = Path(local_html).read_text(encoding="utf-8", errors="ignore")
    content = build_content_from_html(html, page_url=f"local://{local_html}")
    result = {
        "id": content.get("Mã thủ tục", ""),
        "search_code": short_procedure_code(content.get("Mã thủ tục", "")),
        "name": content.get("Tên thủ tục", ""),
        "content": content,
    }
    write_json(out_path, [result])
    print(f"Đã parse thử HTML local và lưu vào: {out_path}")


def run_live(
    input_path: str,
    out_path: str,
    form_dir: str,
    limit: Optional[int] = None,
    headless: bool = False,
    delay: float = 1.0,
) -> None:
    """
    Dùng Playwright vì site mới là React SPA. requests + BeautifulSoup thuần dễ thiếu dữ liệu động
    và không bắt được sự kiện tải file mẫu.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    procedures_list = read_json(input_path)
    if limit:
        procedures_list = procedures_list[:limit]

    form_dir_path = ensure_dir(form_dir)
    out_path = str(out_path)
    ensure_dir(Path(out_path).parent)

    results: List[Dict[str, Any]] = []
    processed = set()

    backup_path = out_path.replace(".json", "_backup.json")
    if Path(backup_path).exists():
        try:
            results = read_json(backup_path)
            processed = {item.get("id") for item in results if item.get("id")}
            print(f"Đã nạp backup: {len(results)} thủ tục.")
        except Exception:
            results = []
            processed = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            accept_downloads=True,
            viewport={"width": 1366, "height": 900},
            locale="vi-VN",
        )
        page = context.new_page()

        total = len(procedures_list)

        for idx, item in enumerate(procedures_list, start=1):
            formal_code = item.get("ma_thu_tuc") or item.get("id") or ""
            old_internal_id = item.get("id", "")
            search_code = short_procedure_code(formal_code)

            if formal_code in processed:
                continue

            print(f"[{idx}/{total}] Đang cào: {formal_code} -> tìm {search_code}")

            try:
                page.goto(BASE_URL, wait_until="networkidle", timeout=60000)

                # Tìm input tìm kiếm chính. Nếu layout đổi nhẹ, ưu tiên input visible đầu tiên.
                inputs = page.locator("input:visible")
                if inputs.count() == 0:
                    raise RuntimeError("Không tìm thấy ô tìm kiếm visible.")

                search_input = inputs.first
                search_input.fill(search_code)

                # Click nút Tìm kiếm hoặc Enter nếu nút không bắt được.
                try:
                    page.get_by_role("button", name=re.compile("Tìm kiếm", re.I)).click(timeout=5000)
                except Exception:
                    search_input.press("Enter")

                page.wait_for_load_state("networkidle", timeout=60000)
                page.wait_for_timeout(1000)

                # Chọn đúng thủ tục trong kết quả.
                link = page.locator(f"a:has-text('{search_code}')").first
                if link.count() == 0:
                    # Một số layout để mã nằm trong text table, thử tìm link trong cùng hàng.
                    row = page.locator(f"tr:has-text('{search_code}')").first
                    link = row.locator("a").first

                if link.count() == 0:
                    raise RuntimeError(f"Không tìm thấy kết quả cho mã {search_code}")

                with page.expect_navigation(wait_until="networkidle", timeout=60000):
                    link.click()

                detail_url = page.url
                page.wait_for_timeout(1000)

                html = page.content()
                content = build_content_from_html(html, page_url=detail_url)

                # Nếu mã thủ tục trên trang chỉ là mã rút gọn, vẫn giữ mã đầy đủ từ danh sách.
                if not content.get("Mã thủ tục"):
                    content["Mã thủ tục"] = search_code

                procedure_name = content.get("Tên thủ tục") or item.get("name") or search_code
                dossier = content.get("Thành phần hồ sơ", [])

                # Tải file mẫu trong mục Thành phần hồ sơ.
                # Mỗi button trong ô mẫu đơn/tờ khai sẽ được click và bắt download event.
                form_order = 0
                try:
                    section = page.locator("xpath=//h4[normalize-space()='Thành phần hồ sơ']/ancestor::div[contains(@class,'bg-white')][1]")
                    rows = section.locator("table tbody tr")
                    row_count = rows.count()

                    for row_idx in range(row_count):
                        row = rows.nth(row_idx)
                        cells = row.locator("td")
                        if cells.count() < 2:
                            continue

                        template_cell = cells.nth(1)
                        original_names_text = template_cell.inner_text(timeout=5000)
                        original_names = split_template_names(original_names_text)
                        buttons = template_cell.locator("button")
                        button_count = buttons.count()

                        # Nếu có text tên mẫu nhưng không có nút tải, vẫn giữ móc nối found_text_only.
                        if button_count == 0:
                            for local_idx, original_name in enumerate(original_names):
                                form_order += 1
                                update_dossier_with_download_result(
                                    dossier,
                                    row_idx,
                                    local_idx,
                                    {
                                        "template_id": f"{formal_code}::M{form_order}",
                                        "order": form_order,
                                        "original_name": original_name,
                                        "stored_name": "",
                                        "relative_path": "",
                                        "download_status": "found_text_only_no_button",
                                        "source_url": detail_url,
                                    },
                                )
                            continue

                        for btn_idx in range(button_count):
                            form_order += 1
                            original_name = original_names[btn_idx] if btn_idx < len(original_names) else ""

                            try:
                                with page.expect_download(timeout=30000) as download_info:
                                    buttons.nth(btn_idx).click()
                                download = download_info.value

                                suggested = download.suggested_filename or original_name or f"mau-{form_order}.bin"
                                stored_name = make_form_filename(search_code, procedure_name, form_order, suggested)
                                saved_path = form_dir_path / stored_name
                                download.save_as(str(saved_path))

                                update_dossier_with_download_result(
                                    dossier,
                                    row_idx,
                                    btn_idx,
                                    {
                                        "template_id": f"{formal_code}::M{form_order}",
                                        "order": form_order,
                                        "original_name": original_name or suggested,
                                        "suggested_filename": suggested,
                                        "stored_name": stored_name,
                                        "relative_path": f"{Path(form_dir).name}/{stored_name}",
                                        "download_status": "downloaded",
                                        "source_url": detail_url,
                                    },
                                )

                            except PlaywrightTimeoutError:
                                update_dossier_with_download_result(
                                    dossier,
                                    row_idx,
                                    btn_idx,
                                    {
                                        "template_id": f"{formal_code}::M{form_order}",
                                        "order": form_order,
                                        "original_name": original_name,
                                        "stored_name": "",
                                        "relative_path": "",
                                        "download_status": "download_timeout",
                                        "source_url": detail_url,
                                    },
                                )
                            except Exception as ex:
                                update_dossier_with_download_result(
                                    dossier,
                                    row_idx,
                                    btn_idx,
                                    {
                                        "template_id": f"{formal_code}::M{form_order}",
                                        "order": form_order,
                                        "original_name": original_name,
                                        "stored_name": "",
                                        "relative_path": "",
                                        "download_status": f"download_error: {ex}",
                                        "source_url": detail_url,
                                    },
                                )

                except Exception as ex:
                    content["download_form_error"] = str(ex)

                content["Thành phần hồ sơ"] = dossier
                content["File mẫu"] = collect_all_forms(dossier)

                record = {
                    "id": formal_code,
                    "old_internal_id": old_internal_id,
                    "search_code": search_code,
                    "name": procedure_name,
                    "detail_url": detail_url,
                    "content": content,
                }

                results.append(record)
                processed.add(formal_code)

                # Backup liên tục để không mất dữ liệu giữa đường.
                save_progress(out_path, results)
                print(f"  OK: {procedure_name} | file mẫu: {len(content.get('File mẫu', []))}")

            except Exception as ex:
                error_record = {
                    "id": formal_code,
                    "old_internal_id": old_internal_id,
                    "search_code": search_code,
                    "name": "",
                    "error": str(ex),
                }
                results.append(error_record)
                processed.add(formal_code)
                save_progress(out_path, results)
                print(f"  LỖI: {formal_code} | {ex}")

            time.sleep(delay)

        browser.close()

    # Ghi file cuối, xóa backup tùy bố muốn giữ. Ở đây giữ backup cho an toàn.
    write_json(out_path, results)
    print(f"Hoàn thành. Đã lưu {len(results)} thủ tục vào {out_path}")
    print(f"File mẫu nằm trong: {form_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--form-dir", default=DEFAULT_FORM_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Chạy ẩn trình duyệt.")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--local-html", default="", help="Parse thử từ file HTML đã lưu, không mở web.")
    args = parser.parse_args()

    if args.local_html:
        parse_local_html(args.local_html, args.out)
        return

    run_live(
        input_path=args.input,
        out_path=args.out,
        form_dir=args.form_dir,
        limit=args.limit,
        headless=args.headless,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
