import pandas as pd


def preprocess_disaster_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare disaster-event data for analysis."""

    data = df.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if "year" in data.columns:
        data["year"] = pd.to_numeric(data["year"], errors="coerce")

    if "latitude" in data.columns:
        data["latitude"] = pd.to_numeric(
            data["latitude"], errors="coerce"
        )

    if "longitude" in data.columns:
        data["longitude"] = pd.to_numeric(
            data["longitude"], errors="coerce"
        )

    data = data.dropna(subset=["year"])

    return data