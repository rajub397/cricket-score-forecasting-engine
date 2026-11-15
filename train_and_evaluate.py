"""
IPL 1st Innings Score Forecasting Engine - Training & Evaluation Pipeline
Uses match-grouped validation (GroupShuffleSplit), ColumnTransformer with
OneHotEncoder + StandardScaler, and RandomForestRegressor for native Termux
and cross-platform compatibility without data leakage.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join("data", "ipl_data.csv")
MODEL_ARTIFACT_PATH = "ipl_score_pipeline.joblib"

CATEGORICAL_FEATURES = ["venue", "bat_team", "bowl_team", "batsman", "bowler"]
NUMERIC_FEATURES = [
    "runs",
    "wickets",
    "overs",
    "runs_last_5",
    "wickets_last_5",
    "striker",
    "non-striker",
]
TARGET_COLUMN = "total"


def load_and_clean_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'. Ensure data/ipl_data.csv exists.")

    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")

    required_cols = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    initial_rows = len(df)
    df = df.dropna(subset=list(required_cols)).copy()
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} incomplete rows.")

    df = df[df["overs"] > 0].reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    regressor = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def main() -> None:
    df = load_and_clean_data(DATA_PATH)

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    if "mid" in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=df["mid"]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(
            f"Match-grouped split complete: {df.iloc[train_idx]['mid'].nunique()} train matches, "
            f"{df.iloc[test_idx]['mid'].nunique()} test matches."
        )
    else:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    pipeline = build_pipeline()
    print("Training Random Forest regression pipeline...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("--------------------------------------------------")
    print(f"Test MAE:  {mae:.2f} runs")
    print(f"Test RMSE: {rmse:.2f} runs")
    print(f"Test R2:   {r2:.4f}")
    print("--------------------------------------------------")

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_cols,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
        },
        MODEL_ARTIFACT_PATH,
    )
    print(f"Saved trained pipeline artifact to '{MODEL_ARTIFACT_PATH}'.")


if __name__ == "__main__":
    main()
