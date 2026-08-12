import pandas as pd

from src.early_warning import assign_risk_level


def test_assign_risk_level():
    df = pd.DataFrame({
        "risk_score": [0.10, 0.50, 0.80, 1.00]
    })

    result = assign_risk_level(df)

    assert list(result["risk_level"]) == [
        "LOW",
        "MEDIUM",
        "HIGH",
        "HIGH",
    ]