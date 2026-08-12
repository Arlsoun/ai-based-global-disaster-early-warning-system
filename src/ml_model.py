from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "disaster_risk_model.joblib"


def train_risk_model(df: pd.DataFrame) -> Pipeline:
    """Train a Random Forest model for disaster risk classification."""

    data = df.copy()

    required_columns = [
        "year",
        "latitude",
        "longitude",
        "risk_score",
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = data.dropna(subset=["risk_score"])

    X = data[["year", "latitude", "longitude"]]
    y = data["risk_score"].astype(int)

    numeric_features = [
        "year",
        "latitude",
        "longitude",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X, y)

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_FILE)

    return model


def predict_risk(
    model: Pipeline,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict risk scores for disaster events."""

    data = df.copy()

    features = data[
        ["year", "latitude", "longitude"]
    ]

    data["predicted_risk_score"] = model.predict(
        features
    )

    return data