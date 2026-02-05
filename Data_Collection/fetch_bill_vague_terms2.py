# -*- coding: utf-8 -*-
# file: count_vague_terms_bills.py
"""
bills_txt 폴더의 모든 .txt에서 '모호표현' 사용을 카운트하여 CSV로 저장.

- 단어 계열(어근/표제어) : Kiwi 토큰의 기본형(form)을 정규식으로 매칭
- 고정 구/표현(n-gram)  : 텍스트 정규식으로 매칭(공백/조사 약간 허용)

출력: bills_vague_term_counts.csv
"""

import os, re, unicodedata, json
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from kiwipiepy import Kiwi

# -----------------------------
# 설정
# -----------------------------
TEXT_DIR = "bills_txt"
OUT_CSV  = "bills_vague_term_counts.csv"

KIWI = Kiwi()

def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s))

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="cp949", errors="ignore") as f:
            return f.read()

# ------------------------------------------------------------------
# 1) 카운트 타깃 정의
#    keys = 사용자 원문 명칭(컬럼명). value = 매칭 전략들(토큰/문자열)
# ------------------------------------------------------------------

# A. 토큰 기반(표제어/어근/품사 조건)
#   - 'lemma_re': Kiwi token.form(기본형)에 적용할 정규식 (NFKC 후)
#   - 'pos_prefix': 선택. 특정 품사 접두(예: 'VA' only 등)로 필터링
TOKEN_RULES = {
    # 형용사/부사 계열(어근 매칭)
    "적절히":      {"lemma_re": r"^적절", "pos_prefix": None},   # 적절히/적절하다/적절한/적절하게/적절성 …
    "충분히":      {"lemma_re": r"^충분", "pos_prefix": None},   # 충분히/충분하다/충분한 …
    "합리적으로":  {"lemma_re": r"^합리", "pos_prefix": None},   # 합리적/합리적인/합리적으로/합리성 …
    "비슷하게":    {"lemma_re": r"^비슷", "pos_prefix": None},   # 비슷한/비슷하게 …
    "드물게":      {"lemma_re": r"^드물", "pos_prefix": None},   # 드문/드물게 …

    # 부사/상태 부류(단일 표제형 매칭)
    "때때로":      {"lemma_re": r"^때때로$", "pos_prefix": None},
    "가끔":        {"lemma_re": r"^가끔$",   "pos_prefix": None},
    "대체로":      {"lemma_re": r"^대체로$", "pos_prefix": None},
    "일반적으로":  {"lemma_re": r"^일반적",  "pos_prefix": None}, # 일반적/일반적으로/일반적인 …
    "보통":        {"lemma_re": r"^보통$",   "pos_prefix": None},
    "주로":        {"lemma_re": r"^주로$",   "pos_prefix": None},
    "종종":        {"lemma_re": r"^종종$",   "pos_prefix": None},
    "간혹":        {"lemma_re": r"^간혹$",   "pos_prefix": None},
    "수시로":      {"lemma_re": r"^수시로$", "pos_prefix": None},
    "정기적으로":  {"lemma_re": r"^정기적",  "pos_prefix": None}, # 정기적/정기적으로 …
    "간헐적으로":  {"lemma_re": r"^간헐적",  "pos_prefix": None},
    "주기적으로":  {"lemma_re": r"^주기적",  "pos_prefix": None},
    "추후":        {"lemma_re": r"^추후$",   "pos_prefix": None},
    "향후":        {"lemma_re": r"^향후$",   "pos_prefix": None},
    "몇몇":        {"lemma_re": r"^몇몇$",   "pos_prefix": None},
    "꽤":          {"lemma_re": r"^꽤$",     "pos_prefix": None},
    "자주":        {"lemma_re": r"^자주$",   "pos_prefix": None},

    # 명사/형용사 ‘일정-’ 중 '일정(스케줄)' 혼동 방지: 형용사(VA*)만 카운트
    "일정하게":    {"lemma_re": r"^일정",    "pos_prefix": "VA"}, # 일정한/일정하게/일정히 …

    # 단일 명사 그대로
    "일부":        {"lemma_re": r"^일부$",   "pos_prefix": None},
}

# B. 구/표현(문자열 정규식)
#    - 공백/조사/어미 약간 허용 (예: '필요 시', '필요시', '필요 시에')
PHRASE_PATTERNS = {
    # 띄어쓰기/조사 허용
    "어느 정도":  r"어느\s*정도",
    "일정 수준":  r"일정\s*수준",
    "필요 시":    r"필요\s*시(에)?|필요시(에)?",
    "때에 따라":  r"때에\s*따라",
    "제때":       r"제때(에)?",
    # 동사구(간단히 텍스트로 포착) — '…할 수도 …'
    "할 수도":    r"\b할\s*수도\b",
}

# 사용자 요청 목록(열 순서 고정)
TARGET_ORDER = [
    "적절히","충분히","일부","일정하게","때때로","가끔","대체로","일반적으로","보통",
    "주로","종종","간혹","드물게","어느 정도","일정 수준","수시로","정기적으로","간헐적으로",
    "주기적으로","필요 시","추후","향후","때에 따라","합리적으로","비슷하게","제때","할 수도",
    "몇몇","꽤","자주",
]

# ------------------------------------------------------------------
# 카운터
# ------------------------------------------------------------------
def count_token_rules(text: str) -> Counter:
    cnt = Counter()
    # 줄 단위로 나눠 인코딩 이슈/메모리 방지
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        L = line.strip()
        if not L:
            continue
        for t in KIWI.tokenize(L):
            lemma = normalize(t.form)
            tag   = t.tag
            for key, rule in TOKEN_RULES.items():
                # 품사 필터
                pp = rule.get("pos_prefix")
                if pp is not None and not tag.startswith(pp):
                    continue
                if re.match(rule["lemma_re"], lemma):
                    cnt[key] += 1
    return cnt

def count_phrase_patterns(text: str) -> Counter:
    cnt = Counter()
    # 전각/반각 통일 + 공백 단순화(연속 공백 -> 1칸)
    T = re.sub(r"\s+", " ", normalize(text))
    for key, pat in PHRASE_PATTERNS.items():
        # overlapped 허용 안 해도 무방; 일반적인 구는 겹치지 않음
        m = re.findall(pat, T)
        if m:
            cnt[key] += len(m)
    return cnt

def process_file(path: str) -> dict:
    text = read_text(path)
    c_tok = count_token_rules(text)
    c_phr = count_phrase_patterns(text)

    # 합치기
    merged = Counter()
    merged.update(c_tok)
    merged.update(c_phr)

    # 결과 행 구성
    row = {"파일": os.path.basename(path)}
    for k in TARGET_ORDER:
        row[k] = int(merged.get(k, 0))
    row["총합"] = int(sum(row[k] for k in TARGET_ORDER))
    return row

def main():
    files = [fn for fn in os.listdir(TEXT_DIR) if fn.endswith(".txt")]
    rows = []
    for fn in tqdm(files, desc="Counting vague terms"):
        rows.append(process_file(os.path.join(TEXT_DIR, fn)))

    df = pd.DataFrame(rows)
    # 열 순서 고정
    df = df[["파일"] + TARGET_ORDER + ["총합"]]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 요약(옵션)
    totals = {k: int(df[k].sum()) for k in TARGET_ORDER}
    with open("bills_vague_term_totals.json", "w", encoding="utf-8") as f:
        json.dump(totals, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료! 저장: {OUT_CSV}")
    print(f"문서 수: {len(files)}")
    print("상위 빈도 상위 10 항목(총합 기준):")
    print(sorted(totals.items(), key=lambda x: -x[1])[:10])

if __name__ == "__main__":
    main()
