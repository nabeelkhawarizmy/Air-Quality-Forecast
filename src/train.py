import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load features and target dataset
IN_PATH = "data/processed/features_and_target_pm25.csv"
if not os.path.exists(IN_PATH):
    raise FileNotFoundError(f"Missing {IN_PATH}. Run src/build_features.py first.")

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Features and target
FEATURES = ["lag_1", "lag_7", "roll_mean_7", "day_of_week", "month"]
TARGET = "y_next_day"

# Basic sanity check
for col in FEATURES + [TARGET]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")


# 2. Split time series into training and testing sets
# Using the most recent 14 days as test, so we don't random split the time series.
TEST_DAYS = 30

if len(df) <= TEST_DAYS + 30:
    # If dataset is small, shrink test size to avoid training on too few rows
    TEST_DAYS = max(7, len(df) // 5)

split_point = len(df) - TEST_DAYS
train_df = df.iloc[:split_point].copy()
test_df = df.iloc[split_point:].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# 3. Baseline model (simple)
# Baseline: "tomorrow ≈ yesterday".
# In features_and_target_pm25.csv, for a row at date t:
#   lag_1 = pm25 at (t-1)
#   y_next_day = pm25 at (t+1)
y_pred_baseline = X_test["lag_1"].values


# 4. Train model using ridge regression
# Standardize the data using StandardScaler first, because the features have different scales. Standardizing helps Ridge regression perform better.
# Pipeline prevents leakage. Flow:
    # X_test → StandardScaler → standardized features
    # Standardized features → Ridge regression → predictions
    # Output: y_pred_ridge (predictions to compare with y_test)
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ]
)

model.fit(X_train, y_train)
y_pred_ridge = model.predict(X_test)


# 5. Evaluate model performance
# FYI: I still find it difficult to understand this step. Need to revisit another time.
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

metrics = {
    "rows_total": int(len(df)),
    "rows_train": int(len(train_df)),
    "rows_test": int(len(test_df)),
    "test_days": int(TEST_DAYS),
    "date_train_min": str(train_df["date"].min().date()),
    "date_train_max": str(train_df["date"].max().date()),
    "date_test_min": str(test_df["date"].min().date()),
    "date_test_max": str(test_df["date"].max().date()),
    "baseline": {
        "definition": "tomorrow ≈ yesterday (uses lag_1)",
        "mae": float(mean_absolute_error(y_test, y_pred_baseline)),
        "rmse": rmse(y_test, y_pred_baseline),
        "r2": float(r2_score(y_test, y_pred_baseline)),
    },
    "ridge": {
        "alpha": 1.0,
        "mae": float(mean_absolute_error(y_test, y_pred_ridge)),
        "rmse": rmse(y_test, y_pred_ridge),
        "r2": float(r2_score(y_test, y_pred_ridge)),
    },
}


# 6. Save artifacts (model and metrics)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

MODEL_PATH = "models/ridge_pm25.joblib"
joblib.dump(model, MODEL_PATH)

PRED_PATH = "reports/predictions.csv"
out = test_df[["date", TARGET]].copy()
out.rename(columns={TARGET: "Actual (next day)"}, inplace=True)
out["Baseline (next day = yesterday)"] = y_pred_baseline
out["Predicted (next day)"] = y_pred_ridge
out.to_csv(PRED_PATH, index=False)

METRICS_PATH = "reports/metrics.json"
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)


# 7. Print summary
print("Saved model to:", MODEL_PATH)
print("Saved predictions to:", PRED_PATH)
print("Saved metrics to:", METRICS_PATH)

print("\n--- Baseline (tomorrow ≈ yesterday) ---")
print("MAE:", metrics["baseline"]["mae"])
print("RMSE:", metrics["baseline"]["rmse"])
print("R2:", metrics["baseline"]["r2"])

print("\n--- Ridge Regression ---")
print("alpha:", metrics["ridge"]["alpha"])
print("MAE:", metrics["ridge"]["mae"])
print("RMSE:", metrics["ridge"]["rmse"])
print("R2:", metrics["ridge"]["r2"])