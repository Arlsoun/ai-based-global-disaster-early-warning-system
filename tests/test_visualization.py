import pandas as pd

from src.visualization import create_risk_distribution_chart


def test_create_risk_distribution_chart():
    df = pd.DataFrame({
        "risk_level": [
            "HIGH",
            "HIGH",
            "MEDIUM",
            "LOW",
        ]
    })

    fig = create_risk_distribution_chart(df)

    assert fig is not None
    assert fig.data
    assert fig.layout.title.text == "Disaster Risk Distribution"