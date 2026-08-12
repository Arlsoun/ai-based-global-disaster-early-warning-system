import pandas as pd

from src.disaster_model import calculate_risk_score


def test_calculate_risk_score():
    df = pd.DataFrame(
        {
            "year": [2020, 2021],
            "latitude": [40.0, 10.0],
            "longitude": [80.0, 20.0],
        }
    )

    result = calculate_risk_score(df)

    assert "risk_score" in result.columns
    assert len(result) == 2
    assert result["risk_score"].iloc[0] == 2
    assert result["risk_score"].iloc[1] == 0