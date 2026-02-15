from openai import OpenAI
import json, re, pandas as pd, glob, os, time, traceback

API_KEY = ""  # ← 여기에 API 키 입력
client = OpenAI(api_key=API_KEY)

PROMPT = """
# [역할 및 지시사항]

당신은 OECD와 EU의 입법 품질 평가 기준에 모두 정통한 10년 경력의 입법 전문 분석가입니다. 당신의 임무는 주어진 법안의 텍스트와 구조를 바탕으로, 아래에 제시된 다차원적인 평가 기준에 따라 그 품질을 객관적으로 분석하고 평가하는 것입니다.

[중요 지침]

1. 객관성 유지: 감정적이거나 정치적인 편향을 완전히 배제하고, 오직 제시된 평가 기준과 법안 텍스트 자체에만 근거하여 논리적으로 평가하십시오.
2. 단계적 사고 (Chain-of-Thought): 각 평가 차원에 대해 점수를 매기기 전에, 왜 그렇게 생각하는지에 대한 구체적인 분석 과정을 먼저 서술해야 합니다. 법안의 특정 조항이나 문구를 직접 인용하여 근거를 제시하십시오.
3. Few-shot 학습: 아래에 제시된 '좋은 품질의 법안 평가 예시'와 '나쁜 품질의 법안 평가 예시'를 숙지하여 평가의 일관된 기준을 학습하십시오.

# [Few-shot 학습 예시]

### <좋은 품질의 법안 평가 예시>

[법안]

- 법안 명: 디지털 취약계층 포용 및 정보격차 해소에 관한 법률안
- 제안이유: 통계청 '2024년 디지털정보격차 실태조사'에 따르면 65세 이상 고령층의 디지털 정보화 수준은 전체 국민 평균의 69.8%에 불과하여... (이하 생략)
- 주요 내용:
    - 제3조(정의) "디지털 취약계층"이란 고령자, 장애인, 저소득층 등...
    - 제5조(국가 및 지방자치단체의 책무) 국가와 지방자치단체는 디지털 취약계층을 위한 맞춤형 교육 프로그램을 개발하고 보급하여야 한다.
    - 제10조(재원 확보) 국가는 본 법의 시행을 위해 필요한 예산을 우선적으로 확보하여야 한다.

[평가 결과]

- 차원 1: 필요성 및 적합성: 9/10점
    - 평가 근거: 제안이유에서 '통계청 실태조사' 결과를 명시적으로 인용하여 법안의 필요성을 객관적 데이터로 입증하고 있다. 제3조에서 "디지털 취약계층"을 명확히 정의하고, 제5조에서 국가의 책무를 구체적으로 규정하여 문제 해결과의 적합성이 높다.
- (이하 평가 생략)
- 최종 종합 점수: 55/60점

### <나쁜 품질의 법안 평가 예시>

[법안]

- 법안 명: 건전한 사회질서 유지를 위한 특별법안
- 제안이유: 최근 사회적으로 부적절한 콘텐츠가 범람하여 올바른 가치관 확립에 심각한 위해를 끼치고 있으므로 이를 바로잡고자 함.
- 주요 내용:
    - 제2조(부적절한 콘텐츠의 규제) 행정기관의 장은 사회 통념에 반하는 부적절한 콘텐츠를 발견한 경우 이를 즉시 삭제하도록 명할 수 있다.
    - 제5조(시행령 위임) 부적절한 콘텐츠의 구체적인 범위와 규제 절차는 대통령령으로 정한다.

[평가 결과]

- 차원 6: 명확성 및 가독성: 3/10점
    - 평가 근거: 제2조의 "사회 통념에 반하는 부적절한 콘텐츠"라는 표현은 매우 모호하고 자의적으로 해석될 소지가 크다. 이는 명확성의 원칙에 위배된다. 또한, 구체적인 규제 범위를 제5조에서 포괄적으로 하위 법령에 위임하고 있어 법률만으로는 국민이 자신의 권리 제한을 예측하기 어렵다.
- (이하 평가 생략)
- 최종 종합 점수: 15/60점

# [본 평가 과업]

아래 법안의 품질을 평가하십시오.

[여기에 평가할 법안의 전문(全文)을 삽입합니다.]

# [평가 기준 및 출력 형식]

다음 6가지 차원에 따라 법안을 평가하고, 아래 출력 형식을 반드시 준수하여 결과를 제시하십시오.

### [평가 기준]

1. 필요성 및 적합성 (Necessity and Relevance): 이 법안이 왜 필요한지, 그리고 해결하려는 문제와 얼마나 잘 부합하는지를 평가합니다.

    - 법안이 해결하고자 하는 국민, 사회의 필요와 우선순위를 명확히 설명하고 있는가?

    - 법안의 목표와 설계가 이러한 필요를 해결하기에 적합하며, 다양한 집단의 필요를 충분히 고려하고 있는가?
    
    - 다른 법률이나 정책으로는 해결할 수 없어 새로운 입법이 반드시 필요한 사안인가?

2. 일관성 및 체계성 (Coherence): 이 법안이 다른 법률이나 정책과 얼마나 조화롭게 작동하는지를 평가합니다.
    - 법안 내 조항들이 서로 모순 없이 논리적으로 일관되는가?
    - 상위 법률(헌법 등)이나 관련된 다른 법률과 충돌하는 부분은 없는가? 다른 정책들과 시너지를 내는가, 상충되는가?
    - 기존 제도와 중복되어 행정력 낭비를 초래할 가능성은 없는가?
3. 효과성 및 비례성 (Effectiveness and Proportionality): 이 법안이 목표를 달성할 수 있을지, 그리고 그 수단이 과도하지 않은지를 평가합니다.
    - 법안이 설정한 목표를 실질적으로 달성할 가능성이 높은가?
    - 규제나 수단의 강도가 해결하려는 문제의 규모에 비례하며, 과도하지 않은가?
    - 법안의 효과가 특정 집단에 편중되지 않고 공정하게 분배되는가?
4. 절차 및 근거 기반 (Process and Evidence Base): 법안이 합리적이고 투명한 과정을 거쳤는지 텍스트를 통해 추론하여 평가합니다.
    - 법안의 제안 이유나 본문에 통계, 연구 결과 등 객관적인 데이터나 근거가 제시되어 있는가?
    - 법안 내용에 이해관계자들의 의견을 수렴하고 폭넓은 사회적 논의를 거친 흔적이 보이는가?
    - 법안의 내용이 일반 국민이 이해하기 쉽고 신뢰할 수 있도록 작성되었는가?
5. 영향력 및 지속가능성 (Impact and Sustainability): 이 법안이 사회에 미칠 장기적, 중대한 영향과 그 효과의 지속 가능성을 평가합니다.
    - 단기적 효과를 넘어 사회 시스템이나 규범에 중대하고 긍정적인(혹은 부정적인) 변화를 가져올 것으로 예상되는가?
    - 의도치 않은 긍정적 또는 부정적 파급 효과는 무엇이 있겠는가?
    - 법안의 긍정적 효과가 재정적, 사회적, 환경적 측면에서 장기적으로 지속될 가능성이 있는가?
6. 명확성 및 가독성 (Clarity and Readability): 법률 조문 자체의 언어적 품질을 평가합니다.
    - 사용된 용어가 명확하고 통일되어 있는가? 모호하거나 다의적으로 해석될 수 있는 표현은 없는가?
    - 문장 구조가 간결하고 이해하기 쉬운가? 하나의 조항에 너무 많은 내용이 복잡하게 얽혀있지는 않은가?
    - 일반 국민이 이 법안을 통해 자신의 권리와 의무를 쉽게 파악할 수 있는가?

[출력 형식(JSON 엄격 준수)]
{
  "dimensions": [
    {"name": "필요성 및 적합성", "score": 1-10, "rationale": "간결한 근거(2~3문장)"},
    {"name": "일관성 및 체계성", "score": 1-10, "rationale": "..."},
    {"name": "효과성 및 비례성", "score": 1-10, "rationale": "..."},
    {"name": "절차 및 근거 기반", "score": 1-10, "rationale": "..."},
    {"name": "영향력 및 지속가능성", "score": 1-10, "rationale": "..."},
    {"name": "명확성 및 가독성", "score": 1-10, "rationale": "..."}
  ],
  "overall_score": 0-60,
  "notes": "선택: 전반적 총평을 2~3문장으로 간결히"
}
"""

PDF_DIR = "./bills_pdfs_2"
OUT_CSV = "./silver_standard_results.csv"
FAIL_LOG = "./failed_files.log"

# ----------------------------
# 유틸: bill_id 추출
# ----------------------------
def extract_bill_id(path: str) -> str:
    m = re.search(r"\[(\d+)\]", path)
    return m.group(1) if m else os.path.basename(path)

# ----------------------------
# 유틸: 응답(raw) → dict 파싱
# ----------------------------
def parse_response_to_dict(raw: str) -> dict:
    try:
        return json.loads(raw)
    except:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise RuntimeError("JSON 블록을 찾지 못함:\n" + raw)
        return json.loads(m.group(0))

# ----------------------------
# 유틸: row(flatten)
# ----------------------------
def build_row(bill_id: str, data: dict) -> dict:
    row = {
        "bill_id"       : bill_id,
        "overall_score" : data.get("overall_score"),
        "notes"         : data.get("notes"),
    }
    for d in data.get("dimensions", []):
        name = d.get("name", "")
        score = d.get("score")
        rationale = d.get("rationale", "")
        base = name.replace(" ", "_")
        row[f"score_{base}"]     = score
        row[f"rationale_{base}"] = rationale
    return row

# ----------------------------
# 유틸: CSV에 안전하게 1행 append
# - 기존 컬럼과 신규 컬럼의 합집합으로 정렬
# - 중복 bill_id 방지
# ----------------------------
def append_row_safely(row: dict, out_csv: str):
    df_row = pd.DataFrame([row])
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        # 이미 처리된 bill_id면 스킵
        if "bill_id" in old.columns and row["bill_id"] in set(old["bill_id"].astype(str)):
            return False  # appended 안 함

        # 컬럼 합집합으로 정렬
        all_cols = list(dict.fromkeys(list(old.columns) + list(df_row.columns)))
        old = old.reindex(columns=all_cols)
        df_row = df_row.reindex(columns=all_cols)
        df = pd.concat([old, df_row], ignore_index=True)
    else:
        df = df_row

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return True  # appended 됨

# ----------------------------
# 유틸: 실패 로그 기록
# ----------------------------
def log_failure(path: str, err: Exception):
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {path}\n{traceback.format_exc()}\n\n")

# ----------------------------
# 메인 루프
# ----------------------------
def main():
    pdf_list = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_list:
        print("PDF가 없습니다:", PDF_DIR)
        return

    # 이미 처리된 bill_id 목록(있다면) 미리 로드하여 건너뛰기
    processed = set()
    if os.path.exists(OUT_CSV):
        try:
            _old = pd.read_csv(OUT_CSV)
            if "bill_id" in _old.columns:
                processed = set(_old["bill_id"].astype(str))
        except Exception:
            pass

    for idx, pdf_path in enumerate(pdf_list, 1):
        bill_id = extract_bill_id(pdf_path)
        if bill_id in processed:
            print(f"[{idx}/{len(pdf_list)}] 이미 처리됨 → 스킵: {pdf_path}")
            continue

        print(f"[{idx}/{len(pdf_list)}] 처리 시작: {pdf_path}")

        # 업로드 + 응답 호출 (재시도 포함)
        retries = 5
        backoff = 2.0
        uploaded_file_id = None

        for attempt in range(1, retries + 1):
            try:
                # 1) 파일 업로드
                up = client.files.create(file=open(pdf_path, "rb"), purpose="user_data")
                uploaded_file_id = up.id

                # 2) 모델 호출
                resp = client.responses.create(
                    model="gpt-5",
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded_file_id},
                            {"type": "input_text", "text": PROMPT}
                        ]
                    }]
                )

                raw = resp.output_text
                data = parse_response_to_dict(raw)
                row = build_row(bill_id, data)

                # 3) CSV에 즉시 반영
                appended = append_row_safely(row, OUT_CSV)
                if appended:
                    processed.add(bill_id)
                    print(f"  → CSV 반영 완료: bill_id={bill_id}")
                else:
                    print(f"  → 중복으로 스킵됨: bill_id={bill_id}")

                break  # 성공 시 루프 탈출

            except Exception as e:
                print(f"  ! 시도 {attempt}/{retries} 실패: {type(e).__name__}: {e}")
                if attempt == retries:
                    print("  → 최종 실패, 로그 기록")
                    log_failure(pdf_path, e)
                else:
                    sleep_s = backoff ** (attempt - 1)
                    time.sleep(sleep_s)

            finally:
                # 업로드 파일 정리(성공/실패 무관)
                if uploaded_file_id:
                    try:
                        client.files.delete(uploaded_file_id)
                    except Exception:
                        # 삭제 실패는 치명적 아님
                        pass

    print("처리 완료. 결과 CSV:", OUT_CSV)
    if os.path.exists(FAIL_LOG):
        print("실패 로그:", FAIL_LOG)

if __name__ == "__main__":
    main()