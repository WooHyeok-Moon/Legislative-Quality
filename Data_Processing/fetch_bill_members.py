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
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp
        except Exception as e:
            logging.warning(f"Fetch failed ({attempt+1}/3): {url} -> {e}")
            sleep(2)
    return None

def extract_bill_title_text(soup):
    """
    상세 페이지에서 전체 법안 제목 텍스트를 최대한 견고하게 추출
    우선순위: h3.detailh3 > h1.tit > p.bill_title
    """
    tag = soup.find("h3", class_="detailh3")
    if not tag:
        tag = soup.find("h1", class_="tit")
    if not tag:
        tag = soup.find("p", class_="bill_title")
    if not tag:
        return None
    # 내부 줄바꿈/공백 정리
    return tag.get_text(" ", strip=True)

def parse_bill_title_for_ids_and_cosponsors(title_text):
    """
    제목 예:
    [2103147] 식품안전기본법 일부개정법률안(강선우의원 등 10인)
    [2111105] 6_25전쟁 ... (한기호의원 등 13인)

    반환: (bill_num, cosponsor_count)
    - bill_num: 대괄호 안 숫자
    - cosponsor_count: '등 xx인'의 xx (int). 없으면 None
    """
    bill_num = None
    cosponsor_count = None

    if title_text:
        m_id = re.search(r'\[(\d+)\]', title_text)
        if m_id:
            bill_num = m_id.group(1)

        # '등 xx인' 패턴 (공백 허용)
        m_co = re.search(r'등\s*([0-9]+)\s*인', title_text)
        if m_co:
            try:
                cosponsor_count = int(m_co.group(1))
            except:
                cosponsor_count = None

    return bill_num, cosponsor_count

def parse_cosponsors_from_page(bill_id: str):
    """
    bill_id 상세 페이지 접속 → 제목 텍스트 확보 → bill_num/공동발의자 수 추출
    """
    url = f"https://likms.assembly.go.kr/bill/bi/billDetailPage.do?billId={bill_id}"
    resp = fetch(url)
    if not resp:
        return None, None, None

    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    title_text = extract_bill_title_text(soup)
    bill_num, cosponsor_count = parse_bill_title_for_ids_and_cosponsors(title_text)

    # 추가 안전장치: 제목 태그를 못 찾았을 때 페이지 전체 텍스트에서 한 번 더 시도
    if title_text is None:
        page_text = soup.get_text(" ", strip=True)
        bill_num2, cosponsor_count2 = parse_bill_title_for_ids_and_cosponsors(page_text)
        bill_num = bill_num or bill_num2
        cosponsor_count = cosponsor_count if cosponsor_count is not None else cosponsor_count2
        title_text = page_text  # 디버깅 확인용

    return bill_num, cosponsor_count, title_text

if __name__ == "__main__":
    # 📂 bill_id.txt 로드
    with open("bill_id.txt", "r", encoding="utf-8") as f:
        bill_ids = [line.strip() for line in f if line.strip()]

    logging.info(f"Loaded {len(bill_ids)} bill IDs")

    rows = []
    for i, bill_id in enumerate(bill_ids, 1):
        bill_num, cos_cnt, title_text = parse_cosponsors_from_page(bill_id)

        rows.append({
            "orig_bill_id": bill_id,        # 원래 PRC... ID
            "bill_num": bill_num,           # [xxxxxxx] 숫자
            "cosponsor_count": cos_cnt,     # 공동발의자 수 (없으면 None)
            "title_text": title_text        # (선택) 원제목: 추출 실패시 디버깅용
        })

        logging.info(f"[{i}/{len(bill_ids)}] {bill_id} -> bill_num={bill_num}, cosponsors={cos_cnt}")
        sleep(uniform(0.8, 1.5))  # 서버 부담 방지

    # CSV 저장 (원하시면 title_text 컬럼은 제거 가능)
    df = pd.DataFrame(rows)
    df.to_csv("bill_cosponsors.csv", index=False, encoding="utf-8-sig")
    logging.info("✅ Saved as bill_cosponsors.csv")
