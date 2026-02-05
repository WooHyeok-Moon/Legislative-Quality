# file: build_vocab_level_all_bills_strict.py
import os, re, unicodedata
import pandas as pd
import numpy as np
from tqdm import tqdm
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# 설정
# -----------------------------
TEXT_DIR = "bills_txt"
LEXICON_CSV = "국어 기초 어휘 선정 및 어휘 등급화 목록 전체.csv"
OUT_SUMMARY_ALL = "bills_vocab_level_strict.csv"
SIM_THRESHOLD = 0.45   # 문맥-뜻풀이 유사도 임계값

KIWI = Kiwi()
SBERT = SentenceTransformer("jhgan/ko-sroberta-multitask")

# -----------------------------
# 유틸
# -----------------------------
def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()

def split_sentences_ending_period(text: str):
    """
    - 줄바꿈을 문장 경계로 취급 (라인 단위 처리)
    - 온점(.)으로 끝나는 문장만 수집
    - 숫자와 붙은 점(날짜/소수/번호 등)은 분할 전에 안전문자(∙)로 치환하여 무시
    - 한국어 문장 종결 필터: 문장은 반드시 [가-힣]. 로 끝나야 하며, 문장 내에 한글이 최소 1개 존재
    - 너무 짧은 조각(길이<=6)은 노이즈로 제외
    """
    sentences = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # 숫자 주변의 점(.)을 안전문자(∙)로 치환하여 분할 대상에서 제외
        safe = line
        # 숫자-점-숫자 (2016.01 / 2016. 01 / 3.14 등)
        safe = re.sub(r'(?<=\d)\.(?=\s*\d)', '∙', safe)
        # 번호식 10. ) / 10. ] / 10. (줄 끝) 등
        safe = re.sub(r'(?<=\d)\.(?=\s*$|\s+[)\]])', '∙', safe)

        start = 0
        for m in re.finditer(r'\.', safe):
            end = m.end()
            chunk = safe[start:end].strip()
            if chunk:
                if len(chunk) <= 6:
                    start = end
                    continue
                if not re.search(r'[가-힣]', chunk):
                    start = end
                    continue
                if not re.search(r'[가-힣]\.$', chunk):
                    start = end
                    continue
                sentences.append(chunk)
            start = end
    return sentences

def map_pos(pos_tag: str) -> str:
    # 분석기 품사 → 엑셀 품사 라벨 단순 매핑
    if pos_tag.startswith("NN"): return "명사"
    if pos_tag.startswith("VV"): return "동사"
    if pos_tag.startswith("VA"): return "형용사"
    if pos_tag.startswith("XSN"): return "접미사"  # ~적 등
    if pos_tag == "NR": return "수사"
    return "기타"

# -----------------------------
# 1) 사전 로드: (표제어, 품사) → [ (sense_id, gloss, grade, gloss_vec) ... ]
# -----------------------------
def load_lexicon(path: str):
    df = pd.read_csv(path)
    # 필수 컬럼 가정: '어휘', '품사', '의미', '등급'
    df["등급수치"] = pd.to_numeric(df["등급"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df = df.dropna(subset=["어휘", "품사", "의미", "등급수치"])

    df["어휘"] = df["어휘"].astype(str).apply(normalize)
    df["품사"] = df["품사"].astype(str).apply(normalize)
    df["의미"] = df["의미"].astype(str).apply(normalize)

    lex = {}
    # 뜻풀이 임베딩은 미리 계산해 캐시(속도 ↑)
    gloss_vecs = SBERT.encode(df["의미"].tolist(), convert_to_tensor=True)
    for (i, r), gvec in zip(df.iterrows(), gloss_vecs):
        key = (r["어휘"], r["품사"])
        gloss = r["의미"]
        grade = float(r["등급수치"])
        lex.setdefault(key, []).append((i, gloss, grade, gvec))
    return lex

# -----------------------------
# 2) 문장 내부 토큰화(표제어)만 수집
# -----------------------------
def tokens_in_sentence(sent: str):
    """
    한 문장(온점으로 끝나는) 내부에서만 표제어/품사 추출.
    반환: [(lemma, pos_label), ...]
    """
    toks = KIWI.tokenize(sent)
    items = []
    for t in toks:
        pos = t.tag
        lemma = t.form  # 기본형
        pos_label = map_pos(pos)
        if pos_label in {"명사", "동사", "형용사", "접미사"}:
            items.append((normalize(lemma), pos_label))
    return items

# -----------------------------
# 3) 후보 의미 선택
# -----------------------------
def pick_grade_from_candidates(lemma: str, pos_label: str, sent_context: str, candidates):
    """
    candidates: [(sense_id, gloss, grade, gloss_vec), ...]
    문장 전체를 문맥으로 사용(문장 경계 밖으로 확장하지 않음)
    """
    if not candidates:
        return None

    if len(candidates) == 1:
        _, _, grade, _ = candidates[0]
        return grade

    ctx_vec = SBERT.encode(sent_context, convert_to_tensor=True)
    best_grade, best_sim = None, -1.0
    for sid, gloss, grade, gloss_vec in candidates:
        sim = float(util.cos_sim(ctx_vec, gloss_vec).item())
        if sim > best_sim:
            best_sim = sim
            best_grade = grade

    # 임계값 미만이면 첫 후보(품사 일치 기준)를 백오프로 선택
    if best_sim < SIM_THRESHOLD:
        _, _, grade, _ = candidates[0]
        return grade
    return best_grade

# -----------------------------
# 4) 단일 파일 처리(요약만 반환)
# -----------------------------
def process_single_file(path: str, lexicon) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        # 인코딩 이슈 대비
        with open(path, "r", encoding="cp949", errors="ignore") as f:
            text = f.read()

    sents = split_sentences_ending_period(text)

    grades = []
    for sent in sents:
        toks = tokens_in_sentence(sent)
        for lemma, pos_label in toks:
            cands = lexicon.get((lemma, pos_label))
            if not cands:
                continue
            g = pick_grade_from_candidates(lemma, pos_label, sent, cands)
            if g is not None:
                grades.append(g)

    avg = float(np.mean(grades)) if grades else np.nan
    return {
        "파일": os.path.basename(path),
        "문장수(온점종결)": len(sents),
        "매칭_토큰수": len(grades),
        "평균_어휘등급": round(avg, 3) if grades else None
    }

# -----------------------------
# 메인
# -----------------------------
def main():
    lex = load_lexicon(LEXICON_CSV)

    rows = []
    files = [fn for fn in os.listdir(TEXT_DIR) if fn.endswith(".txt")]
    for fn in tqdm(files, desc="Processing bills"):
        path = os.path.join(TEXT_DIR, fn)
        row = process_single_file(path, lex)
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT_SUMMARY_ALL, index=False, encoding="utf-8-sig")
    print(f"\n✅ 완료! 요약 CSV 저장: {OUT_SUMMARY_ALL}")
    print(f"총 파일수: {len(rows)}")

if __name__ == "__main__":
    main()
