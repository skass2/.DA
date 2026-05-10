import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import os
import random

BASE_URL = "https://dichvucong.gov.vn/p/home/dvc-tthc-thu-tuc-hanh-chinh-chi-tiet.html?ma_thu_tuc={}&open_popup=1"

# Danh sách một vài User-Agent phổ biến để thay đổi luân phiên
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
]

def get_text(el):
    return el.get_text(separator="\n", strip=True) if el else ""

def parse_fee_block(text, fee_list, lephi_list):
    fee_ids = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line: continue

        if "Lệ phí" in line:
            obj = {"id": f"LP{len(lephi_list)+1}", "text": line}
            lephi_list.append(obj)
            fee_ids.append(obj["id"])
        elif "Mức giá" in line or "Phí" in line:
            obj = {"id": f"P{len(fee_list)+1}", "text": line}
            fee_list.append(obj)
            fee_ids.append(obj["id"])

    return fee_ids

# Tạo một requests Session với cơ chế tự động thử lại (Auto-Retry)
def create_session():
    session = requests.Session()
    # Thử lại tối đa 3 lần, mỗi lần cách nhau một khoảng thời gian tăng dần (backoff_factor)
    retry = Retry(connect=3, backoff_factor=1, status_forcelist=[ 403, 429, 500, 502, 503, 504 ])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def parse_detail(session, internal_id, formal_ma):
    url = BASE_URL.format(internal_id)
    
    # Random User-Agent để ngụy trang
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Connection": "keep-alive"
    }
    
    try:
        # Sử dụng session đã cấu hình retry
        res = session.get(url, headers=headers, timeout=20)
        res.encoding = 'utf-8' 
        soup = BeautifulSoup(res.text, "html.parser")

        modal = soup.select_one(".modal-body")
        if not modal:
            return None

        rows = modal.select(".info-row")
        raw_data = {}

        for row in rows:
            key_el = row.select_one(".key")
            val_el = row.select(".col-sm-9")
            if not key_el or not val_el: continue
            
            key = get_text(key_el).replace(":", "").strip()
            value = get_text(val_el[-1])
            raw_data[key] = value

        content = {
            "Tên thủ tục": raw_data.get("Tên thủ tục", ""),
            "Lĩnh vực": raw_data.get("Lĩnh vực", ""),
            "Đối tượng thực hiện": raw_data.get("Đối tượng thực hiện", ""),
            "Cơ quan thực hiện": raw_data.get("Cơ quan thực hiện", ""),
            "Kết quả thực hiện": raw_data.get("Kết quả thực hiện", ""),
            "Trình tự thực hiện": raw_data.get("Trình tự thực hiện", ""),
            "Yêu cầu điều kiện": raw_data.get("Yêu cầu, điều kiện thực hiện", ""),
            "Cơ quan ban hành": raw_data.get("Cơ quan có thẩm quyền", ""),
            "Cơ quan phối hợp": raw_data.get("Cơ quan phối hợp", "")
        }

        methods, fee_list, lephi_list, hồ_sơ_list, pháp_lý_list = [], [], [], [], []
        tables = modal.select("table.table-data")

        if len(tables) > 0:
            for r in tables[0].select("tbody tr"):
                cols = r.find_all("td")
                if len(cols) >= 4:
                    fee_ids = parse_fee_block(get_text(cols[2]), fee_list, lephi_list)
                    methods.append({
                        "Hình thức": get_text(cols[0]),
                        "Thời hạn": get_text(cols[1]),
                        "Mô tả": get_text(cols[3]),
                        "Liên kết phí": fee_ids
                    })

        content["Cách thức thực hiện"] = methods
        content["Phí"] = fee_list
        content["Lệ phí"] = lephi_list

        if len(tables) > 1:
            for r in tables[1].select("tbody tr"):
                cols = r.find_all("td")
                if len(cols) >= 3:
                    hồ_sơ_list.append({
                        "Tên giấy tờ": get_text(cols[0]), "Biểu mẫu": get_text(cols[1]), "Số lượng": get_text(cols[2])
                    })
        content["Thành phần hồ sơ"] = hồ_sơ_list

        if len(tables) > 2:
            for r in tables[-1].select("tbody tr"):
                cols = r.find_all("td")
                if len(cols) >= 4:
                    pháp_lý_list.append({
                        "Số hiệu": get_text(cols[0]), "Tên văn bản": get_text(cols[1]), "Ngày ban hành": get_text(cols[2]), "Cơ quan ban hành": get_text(cols[3])
                    })
        content["Căn cứ pháp lý"] = pháp_lý_list

        t_ten_thu_tuc = content.get("Tên thủ tục", "")
        if not t_ten_thu_tuc: t_ten_thu_tuc = "Không có thông tin"

        return {
            "id": formal_ma, 
            "name": t_ten_thu_tuc, 
            "content": content
        }

    except Exception as e:
        print(f"❌ {formal_ma} (ID nội bộ: {internal_id}) fail: {e}")
        return None

def run():
    if not os.path.exists("data"): os.makedirs("data")
        
    json_path = "all_laichau_codes.json"
    if not os.path.exists(json_path):
        print(f"❌ Lỗi: Không tìm thấy file '{json_path}'.")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        procedures_list = json.load(f)

    results = []
    
    # ========================================================
    # ĐỌC LẠI FILE BACKUP NẾU CÓ ĐỂ CÀO TIẾP (RESUME)
    # Tránh việc phải cào lại từ đầu những mã đã thành công
    # ========================================================
    processed_ids = set()
    if os.path.exists("data/procedures_backup.json"):
        try:
            with open("data/procedures_backup.json", "r", encoding="utf-8") as fb:
                results = json.load(fb)
                for item in results:
                    processed_ids.add(item["id"])
            print(f"✅ Đã tải lại {len(results)} thủ tục từ file backup. Tiếp tục cào phần còn lại...")
        except:
            pass

    total = len(procedures_list)
    session = create_session() # Khởi tạo Session có retry
    
    print(f"🚀 Bắt đầu cào chi tiết {total} thủ tục...")
    
    for idx, item in enumerate(procedures_list):
        formal_ma = item.get("ma_thu_tuc")
        internal_id = item.get("id")
        
        # Nếu mã này đã cào thành công rồi (có trong file backup) thì bỏ qua
        if formal_ma in processed_ids:
            continue
            
        print(f"🔎 [{idx+1}/{total}] Đang cào: {formal_ma} (Internal ID: {internal_id})")
        
        data = parse_detail(session, internal_id, formal_ma)
        
        if data:
            results.append(data)
            processed_ids.add(formal_ma)
            
        # Lưu backup luỹ tiến
        if len(results) > 0 and len(results) % 10 == 0:
            with open("data/procedures_backup.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        # DELAY AN TOÀN HƠN: Nghỉ từ 1.5s đến 3s thay vì cứng nhắc 1s
        time.sleep(random.uniform(1.5, 3.0)) 

    with open("data/procedures.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    if os.path.exists("data/procedures_backup.json"):
        os.remove("data/procedures_backup.json")

    print(f"\n🎉 HOÀN THÀNH: Đã lưu {len(results)} thủ tục vào data/procedures.json")

if __name__ == "__main__":
    run()