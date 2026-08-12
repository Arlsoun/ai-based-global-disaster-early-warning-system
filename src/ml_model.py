from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "disaster_risk_model.joblib"
)

FEATURE_COLUMNS = [
    "year",
    "latitude",
    "longitude",
]


def train_risk_model(df: pd.DataFrame) -> Pipeline:
    """Train a Random Forest regression model."""

    data = df.copy()

    required_columns = FEATURE_COLUMNS + [
        "risk_score"
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = data.dropna(
        subset=["risk_score"]
    )

    X = data[FEATURE_COLUMNS]

    y = data["risk_score"].astype(float)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                FEATURE_COLUMNS,
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X, y)

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    return model


def predict_risk(
    model: Pipeline,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict continuous disaster risk scores."""

    data = df.copy()

    features = data[FEATURE_COLUMNS]

    data["predicted_risk_score"] = (
        model.predict(features)
    )

    return data


def evaluate_risk_model(
    model: Pipeline,
    df: pd.DataFrame,
) -> dict:
    """Evaluate the disaster risk regression model."""

    required_columns = FEATURE_COLUMNS + [
        "risk_score"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = df.dropna(
        subset=["risk_score"]
    ).copy()

    X = data[FEATURE_COLUMNS]

    y_true = data["risk_score"].astype(float)

    y_pred = model.predict(X)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }