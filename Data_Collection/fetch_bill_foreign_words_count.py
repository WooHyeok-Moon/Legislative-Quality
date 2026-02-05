# file: build_foreign_count_all_bills.py
import os, re, unicodedata, json
from collections import Counter
import pandas as pd
from tqdm import tqdm
from kiwipiepy import Kiwi

# -----------------------------
# 설정
# -----------------------------
TEXT_DIR = "bills_txt"                         # 법안 텍스트 폴더(.txt)
FOREIGN_CSV = "foreign_examples.csv"           # 외국어 단어 목록 (columns: word, example)
OUT_SUMMARY = "bills_foreign_counts.csv"       # 결과 요약 파일
TOPK_PREVIEW = 10                               # 상위 n개 단어 미리보기

KIWI = Kiwi()

# -----------------------------
# 유틸
# -----------------------------
def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip()

def split_lines(text: str):
    """줄 단위 전처리(기존 코드의 문장 분할과 달리, 성능 위해 라인 단위로 토크나이즈)"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [ln.strip() for ln in text.split("\n") if ln.strip()]

def safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="cp949", errors="ignore") as f:
            return f.read()

def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="cp949")

# -----------------------------
# 1) 외국어 단어 목록 로드: set[str]
# -----------------------------
def load_foreign_vocab(path: str):
    df = safe_read_csv(path)
    if "word" not in df.columns:
        raise ValueError("foreign_examples.csv에 'word' 컬럼이 필요합니다.")
    words = set()
    for w in df["word"].astype(str).tolist():
        w = normalize(w)
        if w:
            words.add(w)
    return words

# -----------------------------
# 2) 한 파일 처리: 외국어 표제어 카운트
# -----------------------------
def count_foreign_in_text(text: str, foreign_set: set[str]) -> Counter:
    """
    Kiwi 토크나이저로 표제어(기본형) 단위 일치 카운트.
    - 조사/어미 등은 분리되므로 단어 원형만 정확히 매칭됨.
    - 품사 제한을 두지 않음(리스트의 단어 그대로 매칭).
    """
    cnt = Counter()
    for line in split_lines(text):
        for tok in KIWI.tokenize(line):
            lemma = normalize(tok.form)  # 표면형(기본형). 기존 코드와 동일 사용
            if lemma in foreign_set:
                cnt[lemma] += 1
    return cnt

# -----------------------------
# 3) 메인: 폴더 내 모든 .txt 집계
# -----------------------------
def main():
    foreign_set = load_foreign_vocab(FOREIGN_CSV)

    rows = []
    files = [fn for fn in os.listdir(TEXT_DIR) if fn.endswith(".txt")]
    for fn in tqdm(files, desc="Counting foreign words"):
        path = os.path.join(TEXT_DIR, fn)
        text = safe_read_text(path)
        counts = count_foreign_in_text(text, foreign_set)

        total = sum(counts.values())
        unique = len(counts)
        # 상위 TOPK_PREVIEW 미리보기 문자열 (예: "가드너:3 | 쁘레카:2 | 코아:1")
        top_preview = " | ".join(f"{w}:{c}" for w, c in counts.most_common(TOPK_PREVIEW)) if counts else ""

        # 전체 분포는 JSON 문자열로 저장(원하면 나중에 파싱 가능)
        counts_json = json.dumps(counts, ensure_ascii=False, sort_keys=True)

        rows.append({
            "파일": fn,
            "외국어_총발견건수": total,
            "외국어_고유단어수": unique,
            f"상위{TOPK_PREVIEW}_미리보기": top_preview,
            "전체분포_JSON": counts_json,
        })

    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"\n✅ 완료! 요약 CSV 저장: {OUT_SUMMARY}")
    print(f"총 파일수: {len(rows)}")
    print(f"외국어 단어 수(사전 크기): {len(foreign_set)}")

if __name__ == "__main__":
    main()
