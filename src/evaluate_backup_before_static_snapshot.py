import os
import json
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Part A: Backtest plot (optional)
# -----------------------------
PRED_PATH = "reports/predictions.csv"

df = pd.read_csv(PRED_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Backward-compatible baseline column detection
if "Baseline (next day = yesterday)" in df.columns:
    baseline_col = "Baseline (next day = yesterday)"
elif "Baseline (next day = today)" in df.columns:
    baseline_col = "Baseline (next day = today)"
else:
    raise ValueError("Baseline column not found in predictions.csv")

# Plot actual vs predictions
if os.path.exists(PRED_PATH):
    df = pd.read_csv(PRED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["Predicted (next day)"], label="Predicted (next day)", linewidth=2)
    plt.plot(df["date"], df["Actual (next day)"], label="Actual (next day)", linewidth=2)
    plt.plot(df["date"], df[baseline_col], label=baseline_col, linewidth=2)
    plt.title("Backtest: Next-day PM2.5 Prediction")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    backtest_path = "reports/backtest_plot.png"
    plt.savefig(backtest_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved backtest plot to:", backtest_path)
else:
    print("Backtest plot skipped (no reports/predictions.csv found).")

# -----------------------------
# Part B: 2-day forecast plot (today + tomorrow)
# -----------------------------
SENSOR_TZ = ZoneInfo("Asia/Jakarta")
today_local = datetime.now(SENSOR_TZ).date()
forecast_dates = [today_local, today_local + timedelta(days=1)]

META_PATH = "data/processed/last_observed.json"
DAILY_PATH = "data/processed/daily_pm25.csv"
MODEL_PATH = "models/ridge_pm25.joblib"

for path in [META_PATH, DAILY_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run src/pull_openaq_days.py and src/train.py first.")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

last_observed_date = date.fromisoformat(meta["last_observed_date"])

# Load observed daily PM2.5
df_obs = pd.read_csv(DAILY_PATH)
df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.date
df_obs["pm25"] = pd.to_numeric(df_obs["pm25"], errors="coerce")
df_obs = df_obs.dropna(subset=["date", "pm25"]).sort_values("date").reset_index(drop=True)

# Sanity check: metadata vs CSV
csv_last = df_obs["date"].max()
if csv_last != last_observed_date:
    print(f"Warning: last_observed_date ({last_observed_date}) != CSV max date ({csv_last}). Using CSV max date.")
    last_observed_date = csv_last

# Load trained model
model = joblib.load(MODEL_PATH)

FEATURES = ["lag_1", "lag_7", "roll_mean_7", "day_of_week", "month"]

# Build a date->value map seeded with actuals
pm = {d: float(v) for d, v in zip(df_obs["date"], df_obs["pm25"])}

def make_features(feature_date: date):
    d1 = feature_date - timedelta(days=1)
    d7 = feature_date - timedelta(days=7)

    # Need lag_1 and lag_7 available
    if d1 not in pm or d7 not in pm:
        return None

    # roll_mean_7 uses the previous 7 days: (t-7 ... t-1)
    window = []
    for k in range(1, 8):
        d = feature_date - timedelta(days=k)
        if d not in pm:
            return None
        window.append(pm[d])

    return {
        "lag_1": pm[d1],
        "lag_7": pm[d7],
        "roll_mean_7": sum(window) / 7.0,
        "day_of_week": feature_date.weekday(),
        "month": feature_date.month,
    }

# Recursive forecast until we have values for all forecast_dates
max_needed = max(forecast_dates)

feature_date = last_observed_date
while feature_date + timedelta(days=1) <= max_needed:
    target_date = feature_date + timedelta(days=1)

    # If we already have an actual value for that date, don't overwrite it.
    if target_date in pm:
        feature_date = feature_date + timedelta(days=1)
        continue

    feats = make_features(feature_date)
    if feats is None:
        raise RuntimeError(
            f"Not enough history to build features for {feature_date}. "
            f"Need continuous values for the previous 7 days."
        )

    X = pd.DataFrame([feats])[FEATURES]
    yhat = float(model.predict(X)[0])

    pm[target_date] = yhat
    feature_date = feature_date + timedelta(days=1)

# Save forecast table (today + tomorrow)
rows = []
for d in forecast_dates:
    rows.append(
        {
            "date": d.isoformat(),
            "pm25_forecast": pm[d],
            "steps_ahead_from_last_observed": int((d - last_observed_date).days),
            "last_observed_date": last_observed_date.isoformat(),
            "generated_at_local": datetime.now(SENSOR_TZ).isoformat(),
        }
    )

df_fc = pd.DataFrame(rows)
os.makedirs("reports", exist_ok=True)
out_csv = "reports/forecast_next_2_days.csv"
df_fc.to_csv(out_csv, index=False)
print("Saved forecast table to:", out_csv)

# Plot: last 14 days observed + forecast line up to tomorrow
history_days = 14
cutoff = today_local - timedelta(days=history_days)

df_plot = df_obs[df_obs["date"] >= cutoff].copy()
df_plot["date"] = pd.to_datetime(df_plot["date"])

future_dates = []
future_vals = []
d = last_observed_date + timedelta(days=1)
while d <= max_needed:
    future_dates.append(d)
    future_vals.append(pm[d])
    d += timedelta(days=1)


# ---- BAR CHART ----
fig, ax = plt.subplots(figsize=(14, 6))

# Observed bars
obs_x = mdates.date2num(pd.to_datetime(df_plot["date"]))
obs_y = df_plot["pm25"].astype(float).values
bars_obs = ax.bar(obs_x, obs_y, width=0.8, label="Historical (aggregated daily by OpenAQ)")

# Forecast bars (slightly shifted so it doesn't fully overlap if dates collide)
if future_dates:
    fc_x = mdates.date2num(pd.to_datetime(future_dates))
    fc_y = np.array(future_vals, dtype=float)
    bars_fc = ax.bar(fc_x, fc_y, width=0.8, label="Forecast (may also include yesterday's prediction if not aggregated by OpenAQ yet)", alpha=0.9)
else:
    bars_fc = []

# Put number on each bar
def add_labels(bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:.1f}",
            (b.get_x() + b.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )

add_labels(bars_obs)
if len(bars_fc) > 0:
    add_labels(bars_fc)

ax.set_title("Bogor PM2.5 Air Quality Forecast (Today & Tomorrow)")
ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()

# Show every date on the x-axis
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Ensure x-limits cover the full range neatly
all_dates = list(pd.to_datetime(df_plot["date"]).dt.date.values) + future_dates
xmin = min(all_dates)
xmax = max(all_dates)
ax.set_xlim(mdates.date2num(pd.to_datetime(xmin)) - 1, mdates.date2num(pd.to_datetime(xmax)) + 1)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

forecast_plot_path = "reports/forecast_plot.png"
plt.savefig(forecast_plot_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved forecast plot to:", forecast_plot_path)
print("Last observed day:", last_observed_date.isoformat())
print("Forecasted days:", ", ".join([d.isoformat() for d in forecast_dates]))