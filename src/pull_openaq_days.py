import os
import json
from datetime import date, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv


# Load API key from .env
load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAQ_API_KEY not found. Put it in .env")


# Project-specific constants (simple, explicit)
SENSOR_ID = 13986083

BASE_URL = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/days"
HEADERS = {"X-API-Key": API_KEY}

# Only use completed days (avoid partial-day aggregates)
DATE_TO = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
DATE_FROM = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d")  # ~2 years


# Fetch daily data (simple pagination loop)
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


# Save raw API response (for audit & traceability)
os.makedirs("data/raw", exist_ok=True)

raw_path = f"data/raw/sensor_{SENSOR_ID}_days.json"
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)


# Convert to clean table for modeling
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
            "date": local_start[:10],  # YYYY-MM-DD
            "pm25": r.get("value"),
            "sensor_id": SENSOR_ID,
        }
    )

df = pd.DataFrame(rows)

df["date"] = pd.to_datetime(df["date"])
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)



# Save processed CSV
os.makedirs("data/processed", exist_ok=True)

out_path = "data/processed/daily_pm25.csv"
df.to_csv(out_path, index=False)


# Minimal sanity checks (printed)
print("Saved processed data to:", out_path)
print("Number of days:", len(df))

if len(df) > 0:
    print("Date range:", df["date"].min().date(), "→", df["date"].max().date())
    print("PM2.5 min / max:", df["pm25"].min(), "/", df["pm25"].max())
