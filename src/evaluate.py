from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 1) File paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_CSV = REPO_ROOT / "data" / "processed" / "daily_pm25.csv"
FORECAST_CSV = REPO_ROOT / "reports" / "forecast_fixed_feb_01-14_2026.csv"
OUT_DIR = REPO_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "forecast_plot.png"
OUT_PLOT_DATA = OUT_DIR / "plot_data_snapshot.csv"


# 2) Fixed snapshot window
HIST_START = pd.Timestamp("2026-01-18")
HIST_END = pd.Timestamp("2026-01-31")

FC_START = pd.Timestamp("2026-02-01")
FC_END = pd.Timestamp("2026-02-14")


# 3) Load historical observed data
if not DAILY_CSV.exists():
    raise FileNotFoundError(f"Missing {DAILY_CSV}. Run src/pull_openaq_days.py first.")

df_hist = pd.read_csv(DAILY_CSV)
df_hist["date"] = pd.to_datetime(df_hist["date"])
df_hist["pm25"] = pd.to_numeric(df_hist["pm25"], errors="coerce")
df_hist = df_hist.dropna(subset=["date", "pm25"]).sort_values("date").reset_index(drop=True)

# Filter to Jan 18–31
df_hist = df_hist[(df_hist["date"] >= HIST_START) & (df_hist["date"] <= HIST_END)].copy()


# 4) Load fixed forecast data (Feb 01–14)
if not FORECAST_CSV.exists():
    raise FileNotFoundError(
        f"Missing {FORECAST_CSV}. Run src/forecast_fixed_feb_01-14_2026.py first."
    )

df_fc = pd.read_csv(FORECAST_CSV)
df_fc["date"] = pd.to_datetime(df_fc["date"])

# Accept either column name (old vs new)
if "predicted_pm25" in df_fc.columns:
    pred_col = "predicted_pm25"
elif "pm25_pred" in df_fc.columns:
    pred_col = "pm25_pred"
else:
    raise KeyError(f"Forecast CSV missing prediction column. Found columns: {list(df_fc.columns)}")

df_fc[pred_col] = pd.to_numeric(df_fc[pred_col], errors="coerce")
df_fc = df_fc.dropna(subset=["date", pred_col]).sort_values("date").reset_index(drop=True)

# Filter to Feb 1–14 (safety)
df_fc = df_fc[(df_fc["date"] >= FC_START) & (df_fc["date"] <= FC_END)].copy()


# 5) Combine and save plot data (nice for debugging + transparency)
df_hist_plot = df_hist[["date", "pm25"]].copy()
df_hist_plot["series"] = "Historical (OpenAQ daily)"

df_fc_plot = df_fc[["date", pred_col]].copy()
df_fc_plot.rename(columns={pred_col: "pm25"}, inplace=True)
df_fc_plot["series"] = "Forecast (Fixed Feb 01–14)"

df_plot = pd.concat([df_hist_plot, df_fc_plot], ignore_index=True)
df_plot = df_plot.sort_values("date").reset_index(drop=True)
df_plot.to_csv(OUT_PLOT_DATA, index=False)


# 6) Bar chart
fig, ax = plt.subplots(figsize=(14, 6))

# Historical bars
hist_x = mdates.date2num(df_hist_plot["date"])
hist_y = df_hist_plot["pm25"].astype(float).values
bars_hist = ax.bar(
    hist_x,
    hist_y,
    width=0.8,
    label="Historical (OpenAQ daily aggregates)",
)

# Forecast bars
fc_x = mdates.date2num(df_fc_plot["date"])
fc_y = df_fc_plot["pm25"].astype(float).values
bars_fc = ax.bar(
    fc_x,
    fc_y,
    width=0.8,
    alpha=0.85,
    label="Forecast (February 01–14, 2026)",
)

# Put numbers on top of bars (small text)
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

ax.set_title("Bogor PM2.5 Snapshot: Historical (Jan 18–31) + Forecast (Feb 1–14)")
ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()

# Show every date tick (28 days total, still readable)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Ensure x-limits cover full range neatly
all_dates = list(df_plot["date"].dt.date.values)
xmin = min(all_dates)
xmax = max(all_dates)
ax.set_xlim(
    mdates.date2num(pd.to_datetime(xmin)) - 1,
    mdates.date2num(pd.to_datetime(xmax)) + 1,
)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()

print("Saved plot to:", OUT_PNG)
print("Saved plot data to:", OUT_PLOT_DATA)

# Optional quick summary in terminal
if len(df_hist_plot) > 0:
    print("Historical rows:", len(df_hist_plot), "| Date range:",
          df_hist_plot["date"].min().date(), "→", df_hist_plot["date"].max().date())
else:
    print("Historical rows: 0 (no data found in Jan 18–31 window)")

print("Forecast rows:", len(df_fc_plot), "| Date range:",
      df_fc_plot["date"].min().date(), "→", df_fc_plot["date"].max().date())