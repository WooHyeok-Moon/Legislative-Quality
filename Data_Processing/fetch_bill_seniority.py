import os
import re
import time
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import requests_cache
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# ===================== 설정 =====================
BILL_ID_FILE = "bill_id.txt"          # billId 목록 (한 줄에 하나)
OUTPUT_CSV   = "bill_seniority3.csv"   # 최종 결과
MAX_WORKERS  = 8                      # 의원 개인페이지 병렬 수집 워커 수(5~10 권장)
REQUEST_TIMEOUT = 12                  # 요청 타임아웃
RETRY = 3                             # 재시도 횟수
SLEEP = 0.10                          # 서버 예의상 최소 간격
CACHE_NAME = "assembly_cache"         # 캐시 DB 파일명
CACHE_EXPIRE_SECONDS = 24 * 3600      # 캐시 만료(초) - 24시간
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit(KHTML, like Gecko) Chrome Safari"
    )
}
FIRST_21_BILL_ID = "PRC_K2E0H0I6U0X1T1J1Y3J9R1K8R0L6P0"
# =================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# 캐시 설치 (동일 URL 재요청 시 즉시 응답)
requests_cache.install_cache(
    CACHE_NAME,
    expire_after=CACHE_EXPIRE_SECONDS,
    allowable_methods=["GET"],
    backend="sqlite",
    fast_save=False,  # 병렬 환경에서 안정성↑
)

SESSION = requests.Session()

def safe_path(filename: str) -> str:
    return str(Path(os.getcwd()).joinpath(filename))

def fetch(url: str, *, retry=RETRY, timeout=REQUEST_TIMEOUT, use_cache=True) -> requests.Response:
    """GET + 재시도 + 간격. use_cache=False면 캐시 우회(의원 페이지에 권장)."""
    last_err = None
    for attempt in range(1, retry + 1):
        try:
            if use_cache:
                resp = SESSION.get(url, headers=HEADERS, timeout=timeout)
            else:
                # 캐시 비활성화 (requests_cache 손상/경합 회피)
                with requests_cache.disabled():
                    resp = SESSION.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            if use_cache and not getattr(resp, "from_cache", False):
                time.sleep(SLEEP)
            return resp
        except TypeError as e:
            # 드물게 캐시 역직렬화 오류 방어
            logging.warning(f"[TypeError/cache] {url} ({e}), retry without cache")
            with requests_cache.disabled():
                resp = SESSION.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            time.sleep(0.25 * attempt)
    raise last_err

# ------------------- 1) billDetailPage → 제안자 팝업 링크 파싱 -------------------

def build_proposer_popup_url_from_detail(bill_id: str) -> str:
    """
    법안 상세 페이지에서 '제안자 목록' onclick 파라미터를 파싱해 팝업 URL을 만든다.
    실무상 billId만 넣어도 동작하므로 billId 기준 URL을 우선 사용한다.
    """
    # 1) billDetailPage (새 UI 경로)
    detail_url = f"https://likms.assembly.go.kr/bill/bi/billDetailPage.do?billId={bill_id}"
    resp = fetch(detail_url)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    # onclick="Navigation.bill.popup.billProposer({billId:'...', billNo:'...', ...})" 추출
    a = soup.find("a", class_="icon_list", onclick=re.compile(r"Navigation\.bill\.popup\.billProposer"))
    if a and a.has_attr("onclick"):
        m_id = re.search(r"billId:\s*'([^']+)'", a["onclick"])
        m_no = re.search(r"billNo:\s*'([^']+)'", a["onclick"])
        m_nm = re.search(r"billName:\s*'([^']*)'", a["onclick"])
        bill_no  = m_no.group(1) if m_no else ""
        bill_nm  = m_nm.group(1) if m_nm else ""
        pop_url = (
            "https://likms.assembly.go.kr/bill/bi/popup/billProposer.do?"
            f"billId={bill_id}"
        )
        # billNo/billName이 있으면 붙이고, 없어도 동작함
        if bill_no:
            pop_url += f"&billNo={bill_no}"
        if bill_nm:
            # billName은 이미 escape된 상태일 가능성 → 그대로 전달
            pop_url += f"&billName={bill_nm}"
        return pop_url

    # fallback: billId만으로 접근
    return f"https://likms.assembly.go.kr/bill/bi/popup/billProposer.do?billId={bill_id}"

# ------------------- 2) 제안자 팝업: 페이징 순회 + 의원 링크/정당 -------------------

def proposer_soup(bill_id: str, base_url: str, page: int) -> BeautifulSoup:
    """
    제안자 팝업의 특정 페이지를 반환.
    페이지 파라미터 명이 문서 내 JS에서 숨겨져 있어 추측형으로 시도.
    """
    # 시도할 파라미터 후보 (실무에서 pageIndex가 가장 흔함)
    candidates = ["pageIndex", "pageNo", "page", "currPage", "pageNum"]
    for p in candidates:
        url = f"{base_url}&{p}={page}"
        try:
            resp = fetch(url)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
            # 유효성: 의원 리스트가 있으면 성공
            if soup.select("ul.member_list_img li a[href*='assembly.go.kr/members']"):
                return soup
        except Exception:
            continue
    # 마지막으로 원본(base_url) 그대로(1페이지 추정)
    resp = fetch(base_url)
    resp.encoding = "utf-8"
    return BeautifulSoup(resp.text, "html.parser")

def parse_total_pages(soup: BeautifulSoup) -> int:
    """
    페이지 네비게이션에서 전체 페이지 수를 추론.
    예: <input id="pager_pages_text" value=" (1/2 페이지)">
    """
    # 1) hidden input 우선
    hidden = soup.find("input", id="pager_pages_text")
    if hidden and hidden.has_attr("value"):
        m = re.search(r"/\s*(\d+)\s*페이지", hidden["value"])
        if m:
            return int(m.group(1))
    # 2) 페이지 번호 링크 최대값
    nums = []
    for a in soup.select(".paging-navigation a.number"):
        m = re.search(r"fnSearch\((\d+)\)", a.get("onclick", ""))
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else 1

def collect_members_from_popup(bill_id: str) -> tuple[list[str], list[str], int]:
    """
    제안자 팝업의 모든 페이지를 돌며
    - 의원 개인 페이지 링크들
    - 정당명들
    - 공동발의자 수
    를 반환
    """
    base_url = build_proposer_popup_url_from_detail(bill_id)
    soup1 = proposer_soup(bill_id, base_url, page=1)
    total_pages = parse_total_pages(soup1)

    member_links = []
    parties = []

    def extract(soup: BeautifulSoup):
        # 개인 페이지 링크
        for a in soup.select("ul.member_list_img a[href*='assembly.go.kr/members']"):
            href = a.get("href")
            if href:
                member_links.append(href)
        # 정당
        for p in soup.select("ul.member_list_img p.jdang"):
            parties.append(p.get_text(strip=True))

    extract(soup1)
    # 2페이지 이후
    for p in range(2, total_pages + 1):
        s = proposer_soup(bill_id, base_url, page=p)
        extract(s)

    # 공동발의자 수(중복 제거 전/후 둘 다 의미 있지만, 일반적으로는 링크 개수 사용)
    num_proposers = len(member_links)
    return member_links, parties, num_proposers

# ------------------- 3) 의원 개인 페이지: 당선횟수(선수) -------------------

KOR_SEN_MAP = {
    "초선": 1, "재선": 2, "삼선": 3, "사선": 4, "오선": 5,
    "육선": 6, "칠선": 7, "팔선": 8, "구선": 9, "십선": 10,
}

def parse_member_seniority(member_url: str, base_assembly: int) -> int | None:
    """
    의원 개인 페이지에서 선수를 추출하고,
    base_assembly 기준으로 '미래 대수'는 선수에서 빼서 보정한다.

    - base_assembly == 21:
        제22대가 있으면 -1 (22대는 미래)
    - base_assembly == 20:
        제21대가 있으면 -1, 제22대가 있으면 추가로 -1 (21, 22대 모두 미래)
    """
    try:
        resp = fetch(member_url, use_cache=False)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        text = ""

        # --- 1) 기존 양식 (<dt>당선횟수</dt>) ---
        dt = soup.find("dt", string="당선횟수")
        if dt:
            dd = dt.find_next_sibling("dd")
            if dd:
                text = dd.get_text(" ", strip=True)

        # --- 2) 새 양식 (<div class="jeonzik_left_info">) ---
        if not text:
            # 필요하다면 dl 전체를 가져와도 되고, dt만으로도 "초선(제21대)"는 감지됨
            new_dl = soup.select_one(".jeonzik_left_info .profile_info dl")
            if new_dl:
                text = new_dl.get_text(" ", strip=True)

        if not text:
            return None

        # --- 특정 대수 등장 여부 체크 ---
        has_21 = "제21대" in text
        has_22 = "제22대" in text

        # --- 정규식 기반으로 '현재 페이지 기준' 선수 추출 ---
        seniority = None

        # (1) 숫자 기반 ("4선", "3 선" 등)
        m = re.search(r"(\d+)\s*선", text)
        if m:
            seniority = int(m.group(1))
        else:
            # (2) 한글 기반 ("초선", "재선", "삼선" 등)
            for k, v in KOR_SEN_MAP.items():
                if k in text:
                    seniority = v
                    break

        if seniority is None:
            return None

        # --- base_assembly 기준으로 미래 대수만큼 빼기 ---
        future_terms = 0

        if base_assembly <= 20:
            # 20대 기준이면, 21·22대는 모두 '미래'
            if has_21:
                future_terms += 1
            if has_22:
                future_terms += 1
        elif base_assembly == 21:
            # 21대 기준이면, 22대만 '미래'
            if has_22:
                future_terms += 1
        # base_assembly >= 22 이면 future_terms = 0 (지금은 안 씀)

        adjusted = seniority - future_terms
        # 이론상 1 미만이 나올 일은 없지만, 안전빵으로 최소 1은 보장
        if adjusted < 1:
            adjusted = 1

        return adjusted

    except Exception as e:
        logging.warning(f"[member] fail {member_url} ({e})")
        return None



# ------------------- 4) 단일 bill 처리 -------------------

def process_one_bill(bill_id: str, base_assembly: int) -> dict:
    """
    - 제안자 팝업 전체 페이지 순회
    - 의원별 선수 병렬 수집 (base_assembly 기준 보정)
    - 평균 선수/공동발의자 수/정당 목록 집계
    """
    try:
        member_links, parties, num_proposers = collect_members_from_popup(bill_id)
        unique_parties = sorted(set([p for p in parties if p]))

        # 의원 페이지 병렬 수집
        seniorities = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [
                ex.submit(parse_member_seniority, url, base_assembly)
                for url in member_links
            ]
            for fut in as_completed(futures):
                s = fut.result()
                if isinstance(s, int):
                    seniorities.append(s)

        avg_seniority = round(sum(seniorities) / len(seniorities), 2) if seniorities else None

        return {
            "bill_id": bill_id,
            "num_proposers": num_proposers,
            "avg_seniority": avg_seniority,
            "parties": ", ".join(unique_parties) if unique_parties else None,
        }
    except Exception as e:
        logging.error(f"[{bill_id}] failed: {e}")
        return {
            "bill_id": bill_id,
            "num_proposers": None,
            "avg_seniority": None,
            "parties": None,
            "error": str(e),
        }

# ------------------- 5) 메인 -------------------

def main():
    in_file = Path(BILL_ID_FILE)
    if not in_file.exists():
        raise SystemExit(f"bill_id.txt not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        bill_ids = [ln.strip() for ln in f if ln.strip()]

    # bill_id.txt에서 21대 법안이 처음 등장하는 위치 찾기
    try:
        first21_idx = bill_ids.index(FIRST_21_BILL_ID)
    except ValueError:
        # 혹시 파일에 해당 ID가 없으면, 일단 전부 20대로 간주 (필요시 조정)
        first21_idx = len(bill_ids)

    results = []
    for i, bid in enumerate(tqdm(bill_ids, desc="Bills")):
        # 기준 대수 결정: first21_idx 이전은 20대, 이후는 21대
        base_assembly = 21 if i >= first21_idx else 20
        results.append(process_one_bill(bid, base_assembly))

    df = pd.DataFrame(results)

    out_path = Path(safe_path(OUTPUT_CSV))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    shutil.move(tmp, out_path)

    logging.info(f"✅ Saved {len(df)} rows → {out_path.resolve()}")
    
if __name__ == "__main__":
    main()
