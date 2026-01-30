from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



# ---------- Basic config ----------
st.set_page_config(page_title="Bogor PM2.5 Air Quality Forecast", layout="centered")

BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
DAILY_CSV = BASE_DIR / "data" / "processed" / "daily_pm25.csv"
MODEL_PATH = BASE_DIR / "models" / "ridge_pm25.joblib"

CITY_NAME = "Bogor"
SENSOR_ID = 13986083
WHO_GUIDELINE = 15.0  # simple threshold (your choice)


# ---------- Load artifacts ----------
@st.cache_data
def load_daily_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["date", "pm25"]).sort_values("date").reset_index(drop=True)
    return df


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def compute_features_for_feature_date(history: dict, feature_date: pd.Timestamp) -> dict:
    """
    Matches your training features:

    lag_1: pm25 at (t-1)
    lag_7: pm25 at (t-7)
    roll_mean_7: mean pm25 from (t-7 ... t-1) inclusive (7 values)
    day_of_week: from t
    month: from t

    history: {Timestamp(date)->pm25}, contains observed + predicted values
    feature_date: t
    """
    d = feature_date.normalize()
    d_minus_1 = d - pd.Timedelta(days=1)
    d_minus_7 = d - pd.Timedelta(days=7)

    # We need values for: (t-7..t-1) and (t-7) and (t-1)
    needed_dates = [d - pd.Timedelta(days=i) for i in range(1, 8)]  # t-1 ... t-7
    if any(x not in history for x in needed_dates):
        raise ValueError(f"Not enough history to compute features for {d.date()}")

    lag_1 = history[d_minus_1]
    lag_7 = history[d_minus_7]
    roll_mean_7 = float(np.mean([history[x] for x in needed_dates]))

    return {
        "lag_1": lag_1,
        "lag_7": lag_7,
        "roll_mean_7": roll_mean_7,
        "day_of_week": int(d.dayofweek),
        "month": int(d.month),
    }


def predict_next_day(model, history: dict, feature_date: pd.Timestamp) -> float:
    """
    Predict y_next_day for (feature_date + 1) using features computed at feature_date.
    """
    feats = compute_features_for_feature_date(history, feature_date)
    X = pd.DataFrame([feats], columns=["lag_1", "lag_7", "roll_mean_7", "day_of_week", "month"])
    yhat = float(model.predict(X)[0])
    return yhat


def forecast_until(model, daily_df: pd.DataFrame, target_dates: list[pd.Timestamp]) -> dict:
    """
    Produce predictions for all target_dates by forecasting forward from the last observed day.

    Key idea:
    - We store observed history from daily_df
    - We predict forward day-by-day as needed
    - This handles OpenAQ lag (if last observed < today)

    Returns: {target_date -> predicted_pm25}
    """
    # Build history dict from observed data
    history = {pd.Timestamp(d).normalize(): float(v) for d, v in zip(daily_df["date"], daily_df["pm25"])}

    last_observed = daily_df["date"].max().normalize()
    max_target = max([d.normalize() for d in target_dates])

    predictions = {}

    # We may need to predict intermediate days to reach max_target
    # Our model predicts (t+1) given features at t.
    # So to predict for some day D, we need to run prediction with feature_date = D-1.
    current_feature_date = last_observed

    # Ensure we have enough back-history for first prediction
    # (needs t-7..t-1). If dataset is too short, fail clearly.
    _ = compute_features_for_feature_date(history, current_feature_date)  # will raise if insufficient

    while True:
        next_day = current_feature_date + pd.Timedelta(days=1)

        # Predict next_day using feature_date = current_feature_date
        yhat = predict_next_day(model, history, current_feature_date)

        # Store predicted value into history (so we can go further ahead if needed)
        history[next_day.normalize()] = yhat

        # If this next_day is one of the target dates, capture it
        for td in target_dates:
            if next_day.normalize() == td.normalize():
                predictions[td.normalize()] = yhat

        # Stop if we've reached beyond the latest target date
        if next_day.normalize() >= max_target.normalize():
            break

        # Move forward one day
        current_feature_date = next_day

    return predictions, last_observed


# ---------- UI ----------
st.title("Bogor PM2.5 Air Quality Forecast")
st.caption(f"Bogor, Indonesia • Sensor ID: {SENSOR_ID} • Source: OpenAQ daily aggregates (up to latest completed day)")

# Validate files exist
if not DAILY_CSV.exists():
    st.error(f"Missing file: {DAILY_CSV}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Missing file: {MODEL_PATH}")
    st.stop()

daily_df = load_daily_data(DAILY_CSV)
model = load_model(MODEL_PATH)

last_date = daily_df["date"].max().date()
last_val = float(daily_df.loc[daily_df["date"].idxmax(), "pm25"])

# We want "today" and "tomorrow" in your local system sense
today = pd.Timestamp(date.today())
tomorrow = today + pd.Timedelta(days=1)

# Forecast forward from last observed until we have today and tomorrow
preds, last_observed_ts = forecast_until(model, daily_df, [today, tomorrow])

pred_today = preds.get(today.normalize())
pred_tomorrow = preds.get(tomorrow.normalize())

st.markdown("### Forecast")
c1, c2 = st.columns(2)

with c1:
    st.metric(
        label=f"Today ({today.date()})",
        value=f"{pred_today:.2f} µg/m³" if pred_today is not None else "N/A",
    )
with c2:
    st.metric(
        label=f"Tomorrow ({tomorrow.date()})",
        value=f"{pred_tomorrow:.2f} µg/m³" if pred_tomorrow is not None else "N/A",
    )

# Simple guideline badge using tomorrow prediction (you can change to today)
if pred_tomorrow is not None:
    if pred_tomorrow > WHO_GUIDELINE:
        st.warning(f"**WARNING:** Forecast **above** the {WHO_GUIDELINE} µg/m³ treshold from WHO (World Health Organization).")
    else:
        st.success(f"Forecast **within** the ≤ {WHO_GUIDELINE} µg/m³ treshold from WHO (World Health Organization).")

# Chart: last 14 days actual + forecast points
import matplotlib.dates as mdates  # make sure this import exists near the top

st.markdown("### Last 14 days + forecast")

# Pick a window that's readable as bars (14 is usually nicer than 90 for daily bars)
WINDOW_DAYS = 14
df_plot = daily_df.tail(WINDOW_DAYS).copy()

# Forecast bars data (today + tomorrow)
future_dates = []
future_vals = []

if pred_today is not None:
    future_dates.append(today.normalize())
    future_vals.append(float(pred_today))

if pred_tomorrow is not None:
    future_dates.append(tomorrow.normalize())
    future_vals.append(float(pred_tomorrow))

# ---- BAR CHART ----
fig, ax = plt.subplots(figsize=(14, 6))

# Observed bars
obs_dates = pd.to_datetime(df_plot["date"]).dt.normalize()
obs_x = mdates.date2num(obs_dates)
obs_y = df_plot["pm25"].astype(float).values

bars_obs = ax.bar(
    obs_x,
    obs_y,
    width=0.8,
    label="Historical (aggregated daily by OpenAQ)",
)

# Forecast bars
if future_dates:
    fc_x = mdates.date2num(pd.to_datetime(future_dates))
    fc_y = np.array(future_vals, dtype=float)

    bars_fc = ax.bar(
        fc_x,
        fc_y,
        width=0.8,
        label="Forecast (may also include yesterday's prediction if not aggregated by OpenAQ yet)",
        alpha=0.85,
    )
else:
    bars_fc = []

# Guideline line
ax.axhline(WHO_GUIDELINE, linestyle="--", linewidth=1)
ax.text(
    obs_x.min(),
    WHO_GUIDELINE,
    f" WHO guideline: {WHO_GUIDELINE} µg/m³",
    va="bottom",
    fontsize=9,
)

# Put numbers on bars (optional, but useful)
def add_labels(bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
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

# Title + axes
ax.set_title(f"{CITY_NAME} PM2.5 Air Quality Forecast (Today & Tomorrow)")
ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 level (µg/m³)")
ax.legend()

# X-axis formatting
# For 14 days, showing every day is ok. For larger windows, reduce density.
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Make sure the plot range covers both observed and forecast dates
all_dates = list(obs_dates.dt.date.values) + [d.date() for d in future_dates]
xmin = min(all_dates)
xmax = max(all_dates)
ax.set_xlim(
    mdates.date2num(pd.to_datetime(xmin)) - 1,
    mdates.date2num(pd.to_datetime(xmax)) + 1,
)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

plt.close()

# Small note to prevent confusion
st.caption(
    f"Note: OpenAQ daily aggregates may lag. The latest daily aggregate in this dataset is on {last_observed_ts.date()}. Forecasts are calculated using a ridge regression technique."
)

# Latest observed (completed day)
st.markdown("### Latest observed (completed day)")
st.write(f"**{last_date}** • PM2.5 = **{last_val:.2f} µg/m³**")