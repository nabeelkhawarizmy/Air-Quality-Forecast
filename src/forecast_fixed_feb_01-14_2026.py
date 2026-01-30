from pathlib import Path
from datetime import date
import json

import numpy as np
import pandas as pd
import joblib


# 1) File paths (repo friendly)
REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_CSV = REPO_ROOT / "data" / "processed" / "daily_pm25.csv"
MODEL_PATH = REPO_ROOT / "models" / "ridge_pm25.joblib"
OUT_DIR = REPO_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "forecast_fixed_feb_01-14_2026.csv"
OUT_META = OUT_DIR / "forecast_fixed_feb_01-14_2026_meta.json"


# 2) Fixed forecast window (snapshot)
FORECAST_START = pd.Timestamp(date(2026, 2, 1))
FORECAST_END = pd.Timestamp(date(2026, 2, 14))



# 3) Load inputs
if not DAILY_CSV.exists():
    raise FileNotFoundError(f"Missing {DAILY_CSV}. Run pull_openaq_days.py first.")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing {MODEL_PATH}. Run train.py first.")

# Load data
df = pd.read_csv(DAILY_CSV)
df["date"] = pd.to_datetime(df["date"])
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
df = df.dropna(subset=["date", "pm25"]).sort_values("date").reset_index(drop=True)

# Load model
model = joblib.load(MODEL_PATH)


# 4) Build a "history" dict we can extend with predictions
# history[date] = pm25 value, where date is normalized (00:00)
history = {pd.Timestamp(d).normalize(): float(v) for d, v in zip(df["date"], df["pm25"])}

last_observed = df["date"].max().normalize()


# 5) Feature helpers (must match training)
def compute_features(history_dict: dict, feature_date: pd.Timestamp) -> dict:
    """
    feature_date = t
    model predicts y(t+1) using:
      lag_1 = y(t)
      lag_7 = y(t-6)?? (careful)
    We are using the consistent definition from your pipeline:

    For feature_date t:
      lag_1       = value at (t - 1)
      lag_7       = value at (t - 7)
      roll_mean_7 = mean of (t-1 ... t-7)  (7 values)
      day_of_week = from t
      month       = from t
    """
    t = feature_date.normalize()
    needed = [t - pd.Timedelta(days=i) for i in range(1, 8)]  # t-1 ... t-7

    # If any required past day is missing, we cannot compute features
    for d in needed:
        if d not in history_dict:
            raise ValueError(
                f"Not enough history to compute features for {t.date()} "
                f"(missing {d.date()}). Need at least 7 prior days."
            )

    lag_1 = history_dict[t - pd.Timedelta(days=1)]
    lag_7 = history_dict[t - pd.Timedelta(days=7)]
    roll_mean_7 = float(np.mean([history_dict[d] for d in needed]))

    return {
        "lag_1": lag_1,
        "lag_7": lag_7,
        "roll_mean_7": roll_mean_7,
        "day_of_week": int(t.dayofweek),
        "month": int(t.month),
    }


def predict_next_day(history_dict: dict, feature_date: pd.Timestamp) -> float:
    feats = compute_features(history_dict, feature_date)
    X = pd.DataFrame([feats], columns=["lag_1", "lag_7", "roll_mean_7", "day_of_week", "month"])
    return float(model.predict(X)[0])


# 6) Forecast forward until we cover Feb 14, 2026
# We predict day-by-day starting from "last_observed + 1". This makes the script robust if OpenAQ lags and last_observed < Jan 31.
current_feature_date = last_observed  # we will predict (current_feature_date + 1)

# quick sanity: make sure we can start computing features
_ = compute_features(history, current_feature_date)

forecast_rows = []

# We need predictions up to FORECAST_END
while True:
    next_day = current_feature_date + pd.Timedelta(days=1)

    # Baseline = persistence ("tomorrow = today")
    baseline = history[current_feature_date] if current_feature_date in history else None

    yhat = predict_next_day(history, current_feature_date)

    # Store prediction into history so we can forecast multiple days ahead
    history[next_day.normalize()] = yhat

    # If next_day is inside our fixed forecast window, record it
    if FORECAST_START.normalize() <= next_day.normalize() <= FORECAST_END.normalize():
        forecast_rows.append(
            {
                "date": next_day.date().isoformat(),
                "predicted_pm25": round(yhat, 3),
                "baseline_pm25": round(float(baseline), 3) if baseline is not None else None,
            }
        )

    # Stop once we've predicted through FORECAST_END
    if next_day.normalize() >= FORECAST_END.normalize():
        break

    current_feature_date = next_day


# 7) Save forecast CSV + metadata
out_df = pd.DataFrame(forecast_rows)
out_df.to_csv(OUT_CSV, index=False)

meta = {
    "forecast_start": FORECAST_START.date().isoformat(),
    "forecast_end": FORECAST_END.date().isoformat(),
    "last_observed_date_in_dataset": last_observed.date().isoformat(),
    "notes": "Predictions are generated sequentially from the last observed completed day.",
}
with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Saved:", OUT_CSV)
print("Saved:", OUT_META)
print("Last observed date in dataset:", last_observed.date())
print("Forecast rows:", len(out_df))
if len(out_df) > 0:
    print("Forecast date range:", out_df["date"].iloc[0], "→", out_df["date"].iloc[-1])