import pandas as pd


def calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a simple disaster risk score for each event."""

    data = df.copy()

    if "latitude" in data.columns:
        data["latitude_risk"] = data["latitude"].abs()

    if "longitude" in data.columns:
        data["longitude_risk"] = data["longitude"].abs()

    if "year" in data.columns:
        current_year = data["year"].max()
        data["recency_score"] = current_year - data["year"]

    data["risk_score"] = 0

    if "latitude_risk" in data.columns:
        data["risk_score"] += (
            data["latitude_risk"] > 30
        ).astype(int)

    if "longitude_risk" in data.columns:
        data["risk_score"] += (
            data["longitude_risk"] > 60
        ).astype(int)

    return data