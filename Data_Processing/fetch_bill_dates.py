import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from time import sleep
from random import uniform
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# 요청 함수 (간단한 재시도 포함)
def fetch(url):
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp
        except Exception as e:
            logging.warning(f"Fetch failed ({attempt+1}/3): {url} -> {e}")
            sleep(2)
    return None

def parse_bill_dates(bill_id: str):
    url = f"https://likms.assembly.go.kr/bill/bi/billDetailPage.do?billId={bill_id}"
    resp = fetch(url)
    if not resp:
        return None, None, None, None

    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    # --- (1) 법률안 번호 추출 ---
    title_h3 = soup.find("h3", class_="detailh3")
    numeric_bill_id = None
    if title_h3:
        m = re.search(r'\[(\d+)\]', title_h3.get_text())
        if m:
            numeric_bill_id = m.group(1)  # 괄호 제거한 숫자만

    # --- (2) 제안/의결/결과 파싱 ---
    proposal_date = decision_date = decision_result = None

    for strong_tag in soup.find_all("strong"):
        title = strong_tag.get_text(strip=True)
        div_tag = strong_tag.find_next_sibling("div")
        if not div_tag:
            continue

        if title == "제안일자":
            proposal_date = div_tag.get_text(strip=True)
        elif title == "의결일자":
            decision_date = div_tag.get_text(strip=True)
        elif title == "의결결과":
            decision_result = div_tag.get_text(strip=True)

    return numeric_bill_id, proposal_date, decision_date, decision_result


if __name__ == "__main__":
    # 📂 bill_id.txt 로드
    with open("bill_id.txt", "r", encoding="utf-8") as f:
        bill_ids = [line.strip() for line in f if line.strip()]

    logging.info(f"Loaded {len(bill_ids)} bill IDs")

    data = []

    for i, bill_id in enumerate(bill_ids, 1):
        numeric_id, proposal, decision, result = parse_bill_dates(bill_id)

        data.append({
            "orig_bill_id": bill_id,     # 원래 PRC... ID
            "bill_num": numeric_id,      # [xxxxxxx] ID
            "proposal_date": proposal,
            "decision_date": decision,
            "decision_result": result
        })

        logging.info(f"[{i}/{len(bill_ids)}] {bill_id} -> {numeric_id}: 제안일={proposal}, 의결일={decision}, 결과={result}")
        sleep(uniform(0.8, 1.5))

    # CSV 저장
    df = pd.DataFrame(data)
    df.to_csv("bill_dates.csv", index=False, encoding="utf-8-sig")
    logging.info("✅ Saved as bill_dates.csv")
