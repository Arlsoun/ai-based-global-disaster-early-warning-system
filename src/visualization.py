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
        color="risk_level",
        color_discrete_map={
            "HIGH": "#EF4444",
            "MEDIUM": "#F97316",
            "LOW": "#22C55E",
        },
    )

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        font=dict(color="#F8FAFC"),
        title_font=dict(
            color="#F8FAFC",
            size=18,
        ),
        xaxis=dict(
            title="Risk Level",
            color="#CBD5E1",
            gridcolor="#334155",
        ),
        yaxis=dict(
            title="Number of Events",
            color="#CBD5E1",
            gridcolor="#334155",
        ),
        showlegend=False,
    )

    return fig


def create_disaster_map(df: pd.DataFrame):
    """Create a large geographic map showing disaster-event locations."""

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
        color_discrete_map={
            "HIGH": "#EF4444",
            "MEDIUM": "#F97316",
            "LOW": "#22C55E",
        },
    )

    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        showocean=True,
        fitbounds="locations",
        oceancolor="#08111F",
        landcolor="#172554",
        coastlinecolor="#64748B",
        countrycolor="#334155",
    )

    fig.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        font=dict(color="#F8FAFC"),
        title_font=dict(
            color="#F8FAFC",
            size=18,
        ),
        legend=dict(
            title="Risk Level",
            font=dict(color="#F8FAFC"),
            title_font=dict(color="#F8FAFC"),
        ),
    )

    return fig