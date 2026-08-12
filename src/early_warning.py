import pandas as pd


def assign_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """Assign LOW, MEDIUM, or HIGH risk levels."""

    data = df.copy()

    def get_level(score):
        if score >= 2:
            return "HIGH"
        if score == 1:
            return "MEDIUM"
        return "LOW"

    data["risk_level"] = data["risk_score"].apply(get_level)

    return data