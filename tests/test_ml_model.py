import pandas as pd

from src.data_loader import load_disaster_data
from src.preprocess import preprocess_disaster_data
from src.disaster_model import calculate_risk_score
from src.ml_model import train_risk_model, predict_risk


def test_train_and_predict_risk_model():
    df = load_disaster_data()
    df = preprocess_disaster_data(df)
    df = calculate_risk_score(df)

    sample = df.head(100).copy()

    model = train_risk_model(sample)

    predictions = predict_risk(model, sample)

    assert model is not None
    assert "predicted_risk_score" in predictions.columns
    assert len(predictions) == len(sample)