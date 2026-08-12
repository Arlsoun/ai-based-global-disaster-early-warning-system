import pandas as pd

from src.disaster_model import calculate_risk_score


def test_calculate_risk_score():
    df = pd.DataFrame(
        {
            "country": ["Indonesia", "Japan", "Indonesia"],
            "disastertype": ["flood", "earthquake", "flood"],
            "year": [2020, 2021, 2022],
        }
    )

    result = calculate_risk_score(df)

    assert "risk_score" in result.columns
    assert "country_frequency_score" in result.columns
    assert "disaster_frequency_score" in result.columns
    assert "country_disaster_score" in result.columns
    assert "recency_score" in result.columns

    assert len(result) == 3
    assert result["risk_score"].notna().all()
    assert (result["risk_score"] >= 0).all()