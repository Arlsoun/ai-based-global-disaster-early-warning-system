import pandas as pd

from src.data_loader import load_disaster_data
from src.preprocess import preprocess_disaster_data
from src.disaster_model import calculate_risk_score
from src.ml_model import (
    train_risk_model,
    predict_risk,
    evaluate_risk_model,
)


def prepare_data():
    df = load_disaster_data()
    df = preprocess_disaster_data(df)
    df = calculate_risk_score(df)

    return df.head(100).copy()


def test_train_and_predict_risk_model():
    sample = prepare_data()

    model = train_risk_model(sample)

    predictions = predict_risk(
        model,
        sample,
    )

    assert model is not None
    assert "predicted_risk_score" in predictions.columns
    assert len(predictions) == len(sample)


def test_evaluate_risk_model():
    sample = prepare_data()

    model = train_risk_model(sample)

    evaluation = evaluate_risk_model(
        model,
        sample,
    )

    assert "mae" in evaluation
    assert "rmse" in evaluation
    assert "r2" in evaluation

    assert evaluation["mae"] >= 0
    assert evaluation["rmse"] >= 0
    assert evaluation["r2"] <= 1


def test_risk_predictions_are_numeric():
    sample = prepare_data()

    model = train_risk_model(sample)

    predictions = predict_risk(
        model,
        sample,
    )

    assert pd.api.types.is_numeric_dtype(
        predictions["predicted_risk_score"]
    )