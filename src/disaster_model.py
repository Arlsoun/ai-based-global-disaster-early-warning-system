import pandas as pd


def calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a historical disaster risk score for each event."""

    data = df.copy()

    required_columns = ["country", "disastertype", "year"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    current_year = data["year"].max()

    # Number of events for each country.
    country_frequency = data.groupby("country")["country"].transform("count")

    # Number of events for each disaster type.
    disaster_frequency = data.groupby("disastertype")[
        "disastertype"
    ].transform("count")

    # Number of events for each country and disaster type.
    country_disaster_frequency = data.groupby(
        ["country", "disastertype"]
    )["country"].transform("count")

    # Recency score. More recent events receive higher values.
    years_since_event = current_year - data["year"]
    recency_score = 1 / (1 + years_since_event)

    # Normalize frequency features to 0-1.
    country_score = country_frequency / country_frequency.max()
    disaster_score = disaster_frequency / disaster_frequency.max()
    country_disaster_score = (
        country_disaster_frequency
        / country_disaster_frequency.max()
    )

    # Combine historical frequency and recency.
    data["country_frequency_score"] = country_score
    data["disaster_frequency_score"] = disaster_score
    data["country_disaster_score"] = country_disaster_score
    data["recency_score"] = recency_score

    data["risk_score"] = (
        0.30 * data["country_frequency_score"]
        + 0.30 * data["disaster_frequency_score"]
        + 0.25 * data["country_disaster_score"]
        + 0.15 * data["recency_score"]
    )

    return data