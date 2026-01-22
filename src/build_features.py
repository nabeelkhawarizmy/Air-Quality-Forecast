import os
import pandas as pd

# 1. Load processed daily data
IN_PATH = "data/processed/daily_pm25.csv"
if not os.path.exists(IN_PATH):
    raise FileNotFoundError(f"Missing {IN_PATH}. Run src/pull_openaq_days.py first.")

df = pd.read_csv(IN_PATH)

# Basic cleanup / standardization
if "date" not in df.columns or "pm25" not in df.columns:
    raise ValueError("Input CSV must contain at least 'date' and 'pm25' columns.")

df["date"] = pd.to_datetime(df["date"])
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

# Drop rows with missing essential values
df = df.dropna(subset=["date", "pm25"])

# If there are duplicate dates, keep the first (you could also average them)
df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)


# 2. Create time-based features (no leakage)
# Only use past data
df["lag_1"] = df["pm25"].shift(1)
df["lag_7"] = df["pm25"].shift(7)

# Rolling mean of previous 7 days (exclude current day to avoid leakage)
# Explanation: shift(1) means "up to yesterday", then rolling(7) means "last 7 days"
df["roll_mean_7"] = df["pm25"].shift(1).rolling(window=7).mean()

# Social patterns: Day of week, Month of year
df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
df["month"] = df["date"].dt.month


# 3. Create prediction target (tomorrow)
# y_next_day is the pm25 of the next day (t+1)
df["y_next_day"] = df["pm25"].shift(-1)

# 4. Drop rows that can't be used (missing lags/rolling/target)
feature_cols = ["lag_1", "lag_7", "roll_mean_7", "day_of_week", "month"]
df_model = df.dropna(subset=feature_cols + ["y_next_day"]).copy()

# Keep only what we need (plus date for plotting / debug)
keep_cols = ["date"] + feature_cols + ["y_next_day"]
df_model = df_model[keep_cols].reset_index(drop=True)

# 5. Save feature dataset
OUT_PATH = "data/processed/features_and_target_pm25.csv"
df_model.to_csv(OUT_PATH, index=False)

# 6. Sanity checks (printed)
print("Saved feature dataset to:", OUT_PATH)
print("Rows (usable for ML):", len(df_model))

if len(df_model) > 0:
    print("Feature date range:", df_model["date"].min().date(), "→", df_model["date"].max().date())
    print("Example row (last):")
    print(df_model.tail(1).to_string(index=False))