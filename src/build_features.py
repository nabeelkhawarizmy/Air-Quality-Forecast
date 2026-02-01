import os
import pandas as pd

# 1. Load processed daily data
INPUT_PATH = "data/processed/daily_pm25.csv"
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Missing {INPUT_PATH}. Run src/pull_openaq_days.py first.")
OUTPUT_PATH = "data/processed/features_and_targets_pm25_14d.csv"

HORIZONS = list(range(1, 15))  # 1..14

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])
 # Ensure numeric
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
# If there are duplicate dates, keep the first (you could also average them)
df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
# Drop rows with missing essential values
df = df.dropna(subset=["date", "pm25"])


# 2. Create time-based features (no leakage)
# Lags
df["lag_1"] = df["pm25"].shift(1)
df["lag_2"] = df["pm25"].shift(2)
df["lag_3"] = df["pm25"].shift(3)
df["lag_7"] = df["pm25"].shift(7)
df["lag_14"] = df["pm25"].shift(14)

# Rolling mean of previous 7 days (exclude current day to avoid leakage)
df["roll_mean_7"] = df["pm25"].shift(1).rolling(window=7).mean()
df["roll_mean_14"] = df["pm25"].shift(1).rolling(window=14).mean()
df["roll_std_7"] = df["pm25"].shift(1).rolling(window=7).std()

# Social patterns: Day of week, Month of year
df["dow"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
df["month"] = df["date"].dt.month


# 3. Prediction targets (future values)
for h in HORIZONS:
    df[f"target_h{h:02d}"] = df["pm25"].shift(-h)

feature_cols = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14", "roll_std_7",
    "dow",
    ]

keep_cols = ["date", "pm25"] + feature_cols + [f"target_h{h:02d}" for h in HORIZONS]
df_out = df[keep_cols].copy()


# 4. Drop rows where features are not available yet (early days)
df_out = df_out.dropna(subset=feature_cols).reset_index(drop=True)


# 5. Save feature dataset
os.makedirs("data/processed", exist_ok=True)
df_out.to_csv(OUTPUT_PATH, index=False)


# 6. Sanity checks (printed)
print("Saved:", OUTPUT_PATH)
print("Rows:", len(df_out))
print("Date range:", df_out["date"].min().date(), "→", df_out["date"].max().date())
print("Feature columns:", feature_cols)
print("Targets:", [f"target_h{h:02d}" for h in HORIZONS])