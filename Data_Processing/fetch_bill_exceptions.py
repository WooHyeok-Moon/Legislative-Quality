import os
import re
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# --- 1️⃣ 경로 설정 ---
base_dir = "bills_txt"
output_csv = "bill_exceptions_summary.csv"

# --- 2️⃣ 모델 로드 (KoSBERT 기반 한국어 멀티태스크 모델) ---
print("🔄 모델 불러오는 중...")
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 예외 문장 의미 기준 벡터
example_sentences = [
    "다만 다음 각 호의 어느 하나에 해당하는 경우에는 그러하지 아니하다.",
    "그러나 특별한 사정이 있는 때에는 그러하지 아니하다.",
    "예외적으로 대통령령으로 정하는 경우에는 그러하지 아니하다.",
    "다만 제2항의 경우에는 이를 적용하지 아니한다."
]
example_embeddings = model.encode(example_sentences, convert_to_tensor=True)

# --- 3️⃣ 예외 문장 탐지 함수 ---
def extract_exceptions_from_text(text, threshold=0.45):
    # 온점으로 끝나는 문장만 추출
    sentences = [s.strip() for s in re.findall(r"[^.]+?\.", text)]
    sentences = [s for s in sentences if len(s) > 5]  # 너무 짧은 문장 제외

    # 1차 필터: 키워드 포함 문장만
    keywords = ["다만", "그러나", "예외", "제외", "불구하고"]
    candidates = [
        s for s in sentences
        if any(k in s for k in keywords) and s.endswith(".")
    ]

    results = []
    for s in candidates:
        emb = model.encode(s, convert_to_tensor=True)
        sim = util.cos_sim(emb, example_embeddings).max().item()
        if sim > threshold:
            results.append((s, round(sim, 3)))
    return results

# --- 4️⃣ bills_txt 폴더 내 모든 txt 처리 ---
records = []
files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]

print(f"📂 총 {len(files)}개 법안 처리 중...\n")

for fname in tqdm(files, desc="Processing bills"):
    try:
        # bill_id 추출 (파일명 형태: [2000010] 법안제목.txt)
        m = re.match(r"\[(\d+)\]", fname)
        bill_id = m.group(1) if m else os.path.splitext(fname)[0]

        with open(os.path.join(base_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()

        exceptions = extract_exceptions_from_text(text)
        count = len(exceptions)
        avg_sim = round(sum(sim for _, sim in exceptions) / count, 3) if count > 0 else 0.0
        examples = " | ".join([s for s, _ in exceptions[:3]])  # 대표 문장 3개만 저장

        records.append({
            "bill_id": bill_id,
            "filename": fname,
            "exception_count": count,
            "avg_similarity": avg_sim,
            "examples": examples
        })
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")

# --- 5️⃣ 결과 저장 ---
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ 완료! 총 {len(df)}건의 법안 결과 저장됨.")
print(f"결과 파일: {output_csv}")
