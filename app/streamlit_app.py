import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 1) Snapshot constants (fixed)
HIST_START = pd.Timestamp("2026-01-18")
HIST_END = pd.Timestamp("2026-01-31")
FC_START = pd.Timestamp("2026-02-01")
FC_END = pd.Timestamp("2026-02-14")

CITY_NAME = "Bogor"
SENSOR_ID = 13986083
WHO_GUIDELINE = 15.0

# 2) File paths
REPO_ROOT = Path(__file__).resolve().parents[1]

DAILY_CSV = REPO_ROOT / "data" / "processed" / "daily_pm25.csv"
FORECAST_CSV = REPO_ROOT / "reports" / "forecast_fixed_feb_01-14_2026.csv"
META_JSON = REPO_ROOT / "data" / "processed" / "last_observed.json"
PLOT_PNG = REPO_ROOT / "reports" / "forecast_plot.png"


# 3) Helpers (no caching for safety)
def load_meta():
    if META_JSON.exists():
        with open(META_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_historical():
    if not DAILY_CSV.exists():
        raise FileNotFoundError(f"Missing {DAILY_CSV}. Commit your snapshot CSV to the repo.")

    df = pd.read_csv(DAILY_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["date", "pm25"]).sort_values("date").reset_index(drop=True)

    # Fixed historical window
    df = df[(df["date"] >= HIST_START) & (df["date"] <= HIST_END)].copy()
    return df


def load_forecast():
    if not FORECAST_CSV.exists():
        raise FileNotFoundError(
            f"Missing {FORECAST_CSV}. Run the forecast script and commit the output to the repo."
        )

    df = pd.read_csv(FORECAST_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Support either column name (older vs newer)
    if "predicted_pm25" in df.columns:
        pred_col = "predicted_pm25"
    elif "pm25_pred" in df.columns:
        pred_col = "pm25_pred"
    else:
        raise KeyError(f"Forecast CSV missing prediction column. Found: {list(df.columns)}")

    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df.dropna(subset=["date", pred_col]).sort_values("date").reset_index(drop=True)

    # Fixed forecast window
    df = df[(df["date"] >= FC_START) & (df["date"] <= FC_END)].copy()

    # Standardize name for plotting
    df = df.rename(columns={pred_col: "pm25_pred"})
    return df


def make_bar_chart(df_hist, df_fc):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical bars
    x_hist = mdates.date2num(df_hist["date"])
    y_hist = df_hist["pm25"].astype(float).values
    bars_hist = ax.bar(x_hist, y_hist, width=0.8, label="Historical")

    # Forecast bars
    x_fc = mdates.date2num(df_fc["date"])
    y_fc = df_fc["pm25_pred"].astype(float).values
    bars_fc = ax.bar(x_fc, y_fc, width=0.8, alpha=0.85, label="Forecast")

    # Value labels
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

    add_labels(bars_hist)
    add_labels(bars_fc)

    ax.set_title("Bogor PM2.5 Air Quality: Past 14 days & Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 concentrations (µg/m³)")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a, %d %b"))

    all_dates = list(df_hist["date"].dt.date.values) + list(df_fc["date"].dt.date.values)
    ax.set_xlim(
        mdates.date2num(pd.to_datetime(min(all_dates))) - 1,
        mdates.date2num(pd.to_datetime(max(all_dates))) + 1,
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# 4) Streamlit UI
st.set_page_config(page_title="Bogor Air Quality (PM2.5) Forecast", layout="centered")
st.title("Bogor Air Quality (PM2.5) Forecast")
st.caption(f"Bogor, Indonesia • Sensor ID: {SENSOR_ID} • Data source: OpenAQ daily aggregates")

# Load data
try:
    df_hist = load_historical()
    df_fc = load_forecast()
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Forecast for 01-14 February 2026")

# Prefer showing the pre-rendered plot (consistent with your scripts)
if PLOT_PNG.exists():
    st.image(str(PLOT_PNG), caption="Forecasts are computed using a ridge regression model.\n\n"
    "Data source: OpenAQ daily aggregates.",)
    st.warning(f"**CAUTION:** WHO recommended daily PM2.5 limit is **{WHO_GUIDELINE} µg/m³.**\n\n"
    f"To prevent health risks, reduce time spent outside if it exceeds this limit. We recommend to read the full disclaimer below.")
else:
    fig = make_bar_chart(df_hist, df_fc)
    st.pyplot(fig)
st.markdown("")

# Disclaimers
st.markdown("---")
meta = load_meta()
if meta:
    st.info(
        f"**NOTE:** This Streamlit app's **last GitHub commit was on 01 February 2026**.\n\n"  
        f"However, the webpage may update itself if the app falls to sleep and then woken up by anyone. This is due to Streamlit's Free Community Cloud policy that puts the app to sleep after 12 hours of no traffic."
    )
else:
    st.info("Snapshot mode (static). Metadata file not found, using frozen CSVs only.")

st.markdown("---")

st.subheader("Disclaimers (for transparency)")
st.markdown(
    """
### Prediction model
**Ridge regression model**
- Uses past PM2.5 levels to predict future daily averages. It has a “smoothing” step so it does not overreact to random fluctuations in the training data.
- Learns by computing recent PM2.5 history and calendar patterns:
  - Recent PM2.5 history (e.g. yesterday, last week)
  - 7-day rolling mean (captures weekly rhythm)
  - Day-of-week and month (captures simple seasonal/weekly patterns)
- Training data: OpenAQ daily PM2.5 aggregates (2025-09-05 to 2025-12-29) from an air quality sensor in Bogor.
- Test data: 2025-12-30 to 2025-01-28.
- The cutoff date for commits were 2025-01-31. No live updates were made after the last commit.

### How to interpret this chart
- Blue bars: Observed historical daily averages (18-31 January 2026).
- Orange bars: Model predictions (01-14 February 2026).

### Limitations
- *This forecast is not medical advice.* It is for informational purposes only.
- This model learns from historical data and simple calendar patterns only. It does not compute factors like meteorology (wind, rain, temperature) and holidays/peak seasons.
- Sudden events (e.g. fires, unusual weather) can cause PM2.5 jumps that this model cannot anticipate well.
- OpenAQ aggregates daily average when day is completed. Sensor outages or gaps can affect learnt patterns.
- No minimum performance benchmark shown on this page. A full evaluation/backtest is kept in the repository, but this snapshot view focuses on the frozen forecast only.

### Data & reproducibility (portfolio snapshot)
- This page is intentionally static. The dataset and forecast outputs are frozen to keep the link stable.
- The OpenAQ raw API response is saved in `data/raw/` for traceability.
- The processed data is saved in `data/processed/`.
- Forecast outputs (CSV, plot) are saved in `reports/`.
"""
)

st.caption(
    "Use this forecast as a directional signal. For day-to-day decisions, cross-check with real-time air quality readings."
)

st.markdown("")

# Tables (optional but useful)
st.markdown("### Raw forecast data")
with st.expander("Click to expand"):
    st.write("Historical (Jan 18–31)")
    st.dataframe(df_hist[["date", "pm25"]].reset_index(drop=True), use_container_width=True)

    st.write("Forecast (Feb 1–14)")
    st.dataframe(df_fc[["date", "pm25_pred"]].reset_index(drop=True), use_container_width=True)