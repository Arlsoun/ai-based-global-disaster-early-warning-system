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


def create_disaster_map(df: pd.DataFrame):
    """Create a geographic map showing disaster-event locations."""

    map_data = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    fig = px.scatter_geo(
        map_data,
        lat="latitude",
        lon="longitude",
        color="risk_level",
        hover_name="country",
        hover_data={
            "disastertype": True,
            "year": True,
            "risk_level": True,
            "latitude": False,
            "longitude": False,
        },
        title="Global Disaster Events",
    )

    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        fitbounds="locations",
    )

    return fig