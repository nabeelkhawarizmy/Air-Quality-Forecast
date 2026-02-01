# src/train.py
import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_PATH = "data/processed/features_and_targets_pm25_14d.csv"
MODELS_DIR = "models"

HORIZONS = list(range(1, 15))  # 1..14
TEST_DAYS = 30                 # simple time-split (last 30 days as test)
ALPHA = 1.0                    # Ridge strength (simple default)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing {INPUT_PATH}. Run build_features.py first.")

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
        "roll_mean_7", "roll_mean_14", "roll_std_7",
        "dow",
    ]

    os.makedirs(MODELS_DIR, exist_ok=True)

    metrics = {
        "alpha": ALPHA,
        "test_days": TEST_DAYS,
        "feature_cols": feature_cols,
        "by_horizon": {}
    }

    target_ranges = {}  # to clamp predictions later

    # Time-based split point
    cutoff_date = df["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train_mask_base = df["date"] <= cutoff_date
    test_mask_base = df["date"] > cutoff_date

    for h in HORIZONS:
        target_col = f"target_h{h:02d}"

        # Only keep rows where this horizon's target exists
        df_h = df.dropna(subset=[target_col]).copy()

        train_mask = train_mask_base.loc[df_h.index]
        test_mask = test_mask_base.loc[df_h.index]

        train_df = df_h[train_mask]
        test_df = df_h[test_mask]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=ALPHA)),
        ])

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        metrics["by_horizon"][f"h{h:02d}"] = {
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "date_train_min": str(train_df["date"].min().date()),
            "date_train_max": str(train_df["date"].max().date()),
            "date_test_min": str(test_df["date"].min().date()) if len(test_df) else None,
            "date_test_max": str(test_df["date"].max().date()) if len(test_df) else None,
            "mae": float(mean_absolute_error(y_test, y_pred)) if len(test_df) else None,
            "rmse": float(rmse(y_test, y_pred)) if len(test_df) else None,
            "r2": float(r2_score(y_test, y_pred)) if len(test_df) else None,
        }

        # Save model
        model_path = os.path.join(MODELS_DIR, f"ridge_h{h:02d}.joblib")
        joblib.dump(model, model_path)

        # Save training range (used for optional clamping)
        target_ranges[f"h{h:02d}"] = {
            "train_min": float(y_train.min()),
            "train_max": float(y_train.max())
        }

        print(f"Saved model: {model_path}")

    # Save metrics & ranges
    os.makedirs("reports", exist_ok=True)

    with open("reports/metrics_multi_horizon.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, "target_ranges.json"), "w", encoding="utf-8") as f:
        json.dump(target_ranges, f, indent=2)

    print("Saved: reports/metrics_multi_horizon.json")
    print("Saved:", os.path.join(MODELS_DIR, "target_ranges.json"))


if __name__ == "__main__":
    main()