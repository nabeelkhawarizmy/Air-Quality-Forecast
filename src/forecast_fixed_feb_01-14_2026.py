import os
import json
import joblib
import pandas as pd
import numpy as np

DAILY_PATH = "data/processed/daily_pm25.csv"
LAST_OBS_PATH = "data/processed/last_observed.json"

MODELS_DIR = "models"
RANGES_PATH = os.path.join(MODELS_DIR, "target_ranges.json")

OUT_PATH = "reports/forecast_fixed_feb_01-14_2026.csv"

FORECAST_START = pd.Timestamp("2026-02-01")
FORECAST_END = pd.Timestamp("2026-02-14")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def get_anchor_date(df_daily):
    """
    Prefer cutoff_date from last_observed.json (your portfolio snapshot),
    otherwise fallback to the last date in daily_pm25.csv.
    """
    if os.path.exists(LAST_OBS_PATH):
        with open(LAST_OBS_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "cutoff_date" in meta:
            return pd.to_datetime(meta["cutoff_date"])
    return df_daily["date"].max()


def compute_anchor_features(df_daily, anchor_date):
    """
    Compute features for a SINGLE anchor date using date-based lookups.
    This avoids relying on the 'features file' which may end earlier.
    """
    df = df_daily.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Index by date so we can look up "yesterday", "7 days ago", etc.
    s = df.set_index("date")["pm25"]

    # Normalize timestamp to midnight to match your daily dates
    a = pd.Timestamp(anchor_date).normalize()

    # Required past dates (anchor-1 ... anchor-14)
    needed = [a - pd.Timedelta(days=i) for i in [1, 2, 3, 7, 14]]

    # Check if any needed date is missing
    missing = [d for d in needed if d not in s.index]
    if missing:
        missing_str = ", ".join([str(d.date()) for d in missing])
        raise RuntimeError(
            f"Cannot compute features because these required past dates are missing: {missing_str}\n"
            "This usually happens when the dataset has gaps. "
            "Fix by choosing an anchor date where the previous 14 days exist."
        )

    lag_1 = float(s[a - pd.Timedelta(days=1)])
    lag_2 = float(s[a - pd.Timedelta(days=2)])
    lag_3 = float(s[a - pd.Timedelta(days=3)])
    lag_7 = float(s[a - pd.Timedelta(days=7)])
    lag_14 = float(s[a - pd.Timedelta(days=14)])

    # Rolling windows (based on previous days only)
    last_7 = [a - pd.Timedelta(days=i) for i in range(1, 8)]   # a-1 ... a-7
    last_14 = [a - pd.Timedelta(days=i) for i in range(1, 15)] # a-1 ... a-14

    # Ensure rolling days exist too
    missing7 = [d for d in last_7 if d not in s.index]
    missing14 = [d for d in last_14 if d not in s.index]
    if missing7 or missing14:
        raise RuntimeError(
            "Cannot compute rolling features due to missing days in the last 7/14 days.\n"
            "Pick an anchor date where the previous 14 days all exist."
        )

    roll_mean_7 = float(np.mean([s[d] for d in last_7]))
    roll_mean_14 = float(np.mean([s[d] for d in last_14]))
    roll_std_7 = float(np.std([s[d] for d in last_7], ddof=1))  # sample std

    dow = int(a.dayofweek)

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
        "roll_mean_7", "roll_mean_14", "roll_std_7",
        "dow",
    ]

    X_anchor = pd.DataFrame([{
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "roll_mean_7": roll_mean_7,
        "roll_mean_14": roll_mean_14,
        "roll_std_7": roll_std_7,
        "dow": dow,
    }], columns=feature_cols)

    return X_anchor


def main():
    if not os.path.exists(DAILY_PATH):
        raise FileNotFoundError(f"Missing {DAILY_PATH}. Run pull_openaq_days.py first.")

    df_daily = pd.read_csv(DAILY_PATH)
    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.normalize()
    df_daily["pm25"] = pd.to_numeric(df_daily["pm25"], errors="coerce")
    df_daily = df_daily.dropna(subset=["date", "pm25"]).sort_values("date").reset_index(drop=True)

    # Anchor date (should be 2026-01-31 for Feb 01–14 horizons 1..14)
    anchor_date = get_anchor_date(df_daily).normalize()

    # Horizon check
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    horizons = [(d.normalize() - anchor_date).days for d in forecast_dates]

    if min(horizons) < 1 or max(horizons) > 14:
        raise RuntimeError(
            f"Anchor date is {anchor_date.date()}, which makes horizons {min(horizons)}..{max(horizons)}.\n"
            "To forecast Feb 01–14 with 1..14-day models, anchor date must be 2026-01-31."
        )

    # Compute anchor features directly from daily data
    X_anchor = compute_anchor_features(df_daily, anchor_date)

    # Load clamping ranges (optional)
    target_ranges = {}
    if os.path.exists(RANGES_PATH):
        with open(RANGES_PATH, "r", encoding="utf-8") as f:
            target_ranges = json.load(f)

    preds = []
    for d, h in zip(forecast_dates, horizons):
        model_path = os.path.join(MODELS_DIR, f"ridge_h{h:02d}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model for horizon {h}: {model_path}")

        model = joblib.load(model_path)
        y = float(model.predict(X_anchor)[0])

        # Optional clamping to training min/max (+10% buffer)
        key = f"h{h:02d}"
        if key in target_ranges:
            lo = target_ranges[key]["train_min"]
            hi = target_ranges[key]["train_max"]
            buffer = 0.10 * (hi - lo) if hi > lo else 0.0
            y = clamp(y, lo - buffer, hi + buffer)

        preds.append({
            "date": d.strftime("%Y-%m-%d"),
            "pm25_pred": round(y, 2),
            "horizon_days": h,
        })

    os.makedirs("reports", exist_ok=True)
    out_df = pd.DataFrame(preds)
    out_df.to_csv(OUT_PATH, index=False)

    print("Anchor date used:", anchor_date.date())
    print("Saved:", OUT_PATH)
    print("Forecast rows:", len(out_df))
    print("Forecast date range:", out_df["date"].iloc[0], "→", out_df["date"].iloc[-1])


if __name__ == "__main__":
    main()