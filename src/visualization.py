import pandas as pd
import plotly.express as px


def create_risk_distribution_chart(df: pd.DataFrame):
    """Create a chart showing the number of events by risk level."""

    counts = (
        df["risk_level"]
        .value_counts()
        .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
        .reset_index()
    )

    counts.columns = ["risk_level", "count"]

    fig = px.bar(
        counts,
        x="risk_level",
        y="count",
        title="Disaster Risk Distribution",
        labels={
            "risk_level": "Risk Level",
            "count": "Number of Events",
        },
    )

    return fig