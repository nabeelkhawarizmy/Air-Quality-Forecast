import os
import json
from datetime import date, datetime, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv


# 1) Load API key from .env
load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAQ_API_KEY not found. Put it in .env")

HEADERS = {"X-API-Key": API_KEY}

# 2) Project-specific constants
SENSOR_ID = 13986083
BASE_URL = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/days"

# Freeze the dataset to a fixed cutoff date
CUTOFF_DATE = date(2026, 1, 31)
DATE_TO = CUTOFF_DATE.strftime("%Y-%m-%d")

# Keep enough history for training (2 years)
DATE_FROM = (CUTOFF_DATE - timedelta(days=730)).strftime("%Y-%m-%d")


# 3) Fetch daily data (simple pagination loop)
all_results = []
page = 1
limit = 1000

while True:
    params = {
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "limit": limit,
        "page": page,
    }

    response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAQ API error {response.status_code}: {response.text}"
        )

    payload = response.json()
    results = payload.get("results", [])
    all_results.extend(results)

    # Stop when fewer than `limit` records are returned
    if len(results) < limit:
        break
    page += 1


# 4) Save API response in raw JSON (for audit & traceability)
os.makedirs("data/raw", exist_ok=True)
raw_path = f"data/raw/sensor_{SENSOR_ID}_days.json"
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)


# 5) Convert to clean data for modeling (processed daily CSV)
rows = []
for r in all_results:
    if r.get("parameter", {}).get("name") != "pm25":
        continue

    period = r.get("period", {})
    local_start = period.get("datetimeFrom", {}).get("local")

    if not local_start:
        continue

    rows.append(
        {
            "date": local_start[:10],  # YYYY-MM-DD in +07:00 local time
            "pm25": r.get("value"),
            "sensor_id": SENSOR_ID,
        }
    )

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
df = df.dropna(subset=["date", "pm25"]).sort_values("date").reset_index(drop=True)



# Save processed CSV (hard-freeze again for safety), do not allow any row after cutoff
df = df[df["date"] <= pd.Timestamp(DATE_TO)].copy()

os.makedirs("data/processed", exist_ok=True)
out_path = "data/processed/daily_pm25.csv"
df.to_csv(out_path, index=False)


# 6) Write metadata file (used later by Streamlit)
last_available = df["date"].max().date().isoformat() if len(df) > 0 else None

meta = {
    "sensor_id": SENSOR_ID,
    "cutoff_date": DATE_TO,
    "last_available_date_in_dataset": last_available,
    "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
}
meta_path = "data/processed/last_observed.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)


# 7) Print sanity checks (+ save "last observed day" metadata
print("Saved processed data to:", out_path)
print("Saved metadata to:", meta_path)

print("Cutoff date:", DATE_TO)
print("Number of days:", len(df))
if len(df) > 0:
    print("Date range:", df["date"].min().date(), "→", df["date"].max().date())
    print("PM2.5 min / max:", float(df["pm25"].min()), "/", float(df["pm25"].max()))
else:
    print("No PM2.5 daily rows returned. Check SENSOR_ID, API key, or date window.")