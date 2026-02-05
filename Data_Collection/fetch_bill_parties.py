import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm

# === 1️⃣ bill_id.txt 불러오기 ===
with open("bill_id.txt", "r", encoding="utf-8") as f:
    bill_ids = [line.strip() for line in f if line.strip()]

# === 2️⃣ 결과 저장용 ===
results = []

# === 3️⃣ 각 bill_id 순회 ===
for bill_id in tqdm(bill_ids, desc="Processing bills"):
    try:
        # 핵심: 실제 작동하는 URL 패턴
        url = f"https://likms.assembly.go.kr/bill/bi/popup/billProposer.do?billId={bill_id}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/118.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "utf-8"
        
        if resp.status_code != 200:
            print(f"[!] 접근 실패 ({resp.status_code}): {bill_id}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        parties = [p.get_text(strip=True) for p in soup.select("p.jdang")]
        unique_parties = sorted(set(parties))

        results.append({
            "bill_id": bill_id,
            "num_parties": len(unique_parties),
            "parties": ", ".join(unique_parties)
        })

        time.sleep(0.5)

    except Exception as e:
        print(f"[ERROR] {bill_id}: {e}")

# === 4️⃣ CSV 저장 ===
df = pd.DataFrame(results)
df.to_csv("bill_party_counts.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ 완료! 총 {len(df)}개의 법안 처리됨.")
print("결과 파일: bill_party_counts.csv")