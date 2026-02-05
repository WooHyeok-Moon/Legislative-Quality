import os
import re
import pandas as pd
from tqdm import tqdm

# 📂 법안 텍스트 폴더 경로
TEXT_DIR = "bills_txt"

# ⚙️ 단어 탐색용 정규식 (한글, 영어, 숫자 포함)
word_pattern = re.compile(r"\b[\w가-힣]+\b")

def split_sentences_ending_period(text: str):
    """
    - 줄바꿈을 문장 경계로 취급 (라인 단위 처리)
    - 온점(.)으로 끝나는 문장만 수집
    - 숫자와 붙은 점(날짜/소수/번호 등)은 분할 전에 안전문자(∙)로 치환하여 무시
    - 한국어 문장 종결 필터: 문장은 반드시 [가-힣]. 로 끝나야 하며, 문장 내에 한글이 최소 1개 존재
    - 너무 짧은 조각(길이<=6)은 노이즈로 제외
    """
    sentences = []

    # 개행 정규화
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # 1) 숫자 주변의 점(.)을 안전문자(∙)로 치환하여 분할 대상에서 제외
        safe = line
        # (a) 2016. 01 / 2016.01 / 3.14 등 숫자-점-숫자
        safe = re.sub(r'(?<=\d)\.(?=\s*\d)', '∙', safe)
        # (b) 번호식 10. ) / 10. ] / 10. (줄 끝) 등
        safe = re.sub(r'(?<=\d)\.(?=\s*$|\s+[)\]])', '∙', safe)

        # 2) 남은 온점(.) 기준으로 같은 라인 내에서만 문장 분리
        start = 0
        for m in re.finditer(r'\.', safe):
            end = m.end()
            chunk = safe[start:end].strip()

            # 원본 라인에서 같은 구간을 복원(치환된 ∙는 다시 .로 돌릴 필요 없음)
            if chunk:
                # 필터 1: 너무 짧은 조각 버리기
                if len(chunk) <= 6:
                    start = end
                    continue
                # 필터 2: 반드시 한글 포함
                if not re.search(r'[가-힣]', chunk):
                    start = end
                    continue
                # 필터 3: 한국어 종결 조건 - 마침표 직전이 한글
                if not re.search(r'[가-힣]\.$', chunk):
                    start = end
                    continue

                sentences.append(chunk)

            start = end

        # 라인 끝에 남은 텍스트는 온점으로 끝나지 않으면 버림(요구사항)
    return sentences


def calc_avg_words_per_sentence(filepath):
    """
    파일에서 '온점으로 끝나는 한국어 문장'만 문장으로 간주하고
    각 문장의 단어 수 평균 계산
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # 안전한 문장 분리
        sentences = split_sentences_ending_period(text)

        if not sentences:
            return 0.0  # 온점으로 끝나는 유효 문장이 없는 경우

        # 각 문장의 단어 수 계산
        word_counts = []
        for s in sentences:
            words = word_pattern.findall(s)
            word_counts.append(len(words))

        # 평균 계산
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        return round(avg_words, 2)

    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return 0.0


if __name__ == "__main__":
    results = []

    for filename in tqdm(os.listdir(TEXT_DIR), desc="Analyzing sentence length"):
        if filename.endswith(".txt"):
            bill_id = os.path.splitext(filename)[0]
            path = os.path.join(TEXT_DIR, filename)

            avg_len = calc_avg_words_per_sentence(path)
            results.append({
                "bill_id": bill_id,
                "avg_words_per_sentence": avg_len
            })

    # CSV로 저장
    df = pd.DataFrame(results)
    df.to_csv("bill_avg_sentence_length.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 완료! 'bill_avg_sentence_length.csv' 파일이 생성되었습니다.")
