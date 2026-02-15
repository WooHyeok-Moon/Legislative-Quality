import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from time import sleep
from random import uniform
import re

# --------------------------------------
# 설정
# --------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE = "https://likms.assembly.go.kr"
DETAIL_URL = BASE + "/bill/bi/billDetailPage.do"
BILLINFO_URL = BASE + "/bill/bi/bill/detail/billInfo.do"

HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.7,en;q=0.6",
}

AJAX_HEADERS = {
    **HDRS,
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}

BILLNUM_REGEX = re.compile(r"\[(\d+)\]")
SECTIONS = ["소관위 회의정보", "법사위 회의정보", "본회의 심의정보"]

# --------------------------------------
# 유틸 함수
# --------------------------------------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def get_bill_number_from_detail(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    h3 = soup.select_one("h3.detailh3")
    if not h3:
        return ""
    text = h3.get("title") or h3.get_text(" ", strip=True)
    m = BILLNUM_REGEX.search(text)
    return m.group(1) if m else ""

def find_section_table_by_h4(soup: BeautifulSoup, h4_contains: str):
    """h4 텍스트나 caption에 특정 문구가 포함된 table을 찾음"""
    for h4 in soup.find_all("h4"):
        if h4_contains in normalize_space(h4.get_text(" ", strip=True)):
            div = h4.find_parent("div")
            if not div:
                continue
            table = div.find("table")
            if table:
                return table
    # fallback: caption 기반
    for t in soup.find_all("table"):
        cap = t.find("caption")
        if cap and h4_contains in normalize_space(cap.get_text(" ", strip=True)):
            return t
    return None

def extract_titles_from_billinfo_html(html: str):
    """billInfo.do 응답 HTML에서 회의명 목록 추출"""
    soup = BeautifulSoup(html, "html.parser")
    titles = []
    for label in SECTIONS:
        table = find_section_table_by_h4(soup, label)
        if not table:
            continue
        for span in table.select("tbody td.mtngName > span"):
            t = normalize_space(span.get_text())
            if t:
                titles.append(t)
    return titles

def fetch_meeting_titles_for_bill(bill_id: str):
    """법안별 회의명 수집"""
    with requests.Session() as sess:
        # 1️⃣ 상세 페이지 접근: 쿠키 세팅 + 법안번호 추출
        r = sess.get(DETAIL_URL, params={"billId": bill_id}, headers=HDRS, timeout=15)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        bill_no = get_bill_number_from_detail(r.text)

        # 2️⃣ billInfo.do XHR POST
        headers = {**AJAX_HEADERS, "Referer": f"{DETAIL_URL}?billId={bill_id}"}
        resp = sess.post(BILLINFO_URL, data={"billId": bill_id}, headers=headers, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        titles = extract_titles_from_billinfo_html(resp.text)
        return bill_no, titles, resp.text

# --------------------------------------
# 실행부
# --------------------------------------
if __name__ == "__main__":
    with open("bill_id.txt", "r", encoding="utf-8") as f:
        bill_ids = [line.strip() for line in f if line.strip()]

    rows = []
    for i, bid in enumerate(bill_ids, 1):
        try:
            bill_no, titles, html = fetch_meeting_titles_for_bill(bid)

            if not titles:
                # 회의정보가 없을 경우 HTML 저장
                with open(f"debug_billinfo_{bid}.html", "w", encoding="utf-8") as df:
                    df.write(html)

            rows.append({
                "bill_id": bid,
                "법안번호": bill_no,
                "회의명": " | ".join(titles),
                "회의명개수": len(titles),
            })

            logging.info(f"[{i}/{len(bill_ids)}] {bid} -> {len(titles)} titles")

        except Exception as e:
            logging.warning(f"[{i}/{len(bill_ids)}] {bid} ERROR: {e}")
            rows.append({
                "bill_id": bid,
                "법안번호": "",
                "회의명": "",
                "회의명개수": 0,
            })

        sleep(uniform(0.5, 1.2))  # 서버 부담 방지

    df = pd.DataFrame(rows)
    df.to_csv("meeting_titles.csv", index=False, encoding="utf-8-sig")
    logging.info("✅ Saved meeting_titles.csv")
