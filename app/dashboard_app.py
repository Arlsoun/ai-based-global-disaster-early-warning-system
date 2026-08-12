import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_disaster_data
from src.preprocess import preprocess_disaster_data
from src.disaster_model import calculate_risk_score
from src.ml_model import train_risk_model, predict_risk
from src.early_warning import assign_risk_level
from src.visualization import (
    create_risk_distribution_chart,
    create_disaster_map,
)

st.set_page_config(
    page_title="AI-Based Global Disaster Early-Warning System",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 AI-Based Global Disaster Early-Warning System")
st.caption("Disaster-event analysis and early-warning risk assessment")

try:
    df = load_disaster_data()
    df = preprocess_disaster_data(df)
    df = calculate_risk_score(df)

    model = train_risk_model(df)
    df = predict_risk(model, df)

    df = assign_risk_level(df)

    st.success("Disaster data loaded successfully.")

    st.sidebar.header("Dashboard Filters")

    countries = sorted(
        df["country"].dropna().unique().tolist()
    )

    disaster_types = sorted(
        df["disastertype"].dropna().unique().tolist()
    )

    risk_levels = ["HIGH", "MEDIUM", "LOW"]

    selected_countries = st.sidebar.multiselect(
        "Country",
        options=countries,
        default=[],
    )

    selected_disaster_types = st.sidebar.multiselect(
        "Disaster Type",
        options=disaster_types,
        default=[],
    )

    selected_risk_levels = st.sidebar.multiselect(
        "Risk Level",
        options=risk_levels,
        default=[],
    )

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    selected_years = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    filtered_df = df.copy()

    if selected_countries:
        filtered_df = filtered_df[
            filtered_df["country"].isin(selected_countries)
        ]

    if selected_disaster_types:
        filtered_df = filtered_df[
            filtered_df["disastertype"].isin(
                selected_disaster_types
            )
        ]

    if selected_risk_levels:
        filtered_df = filtered_df[
            filtered_df["risk_level"].isin(
                selected_risk_levels
            )
        ]

    filtered_df = filtered_df[
        filtered_df["year"].between(
            selected_years[0],
            selected_years[1],
        )
    ]

    high_risk_count = int(
        (filtered_df["risk_level"] == "HIGH").sum()
    )

    medium_risk_count = int(
        (filtered_df["risk_level"] == "MEDIUM").sum()
    )

    low_risk_count = int(
        (filtered_df["risk_level"] == "LOW").sum()
    )

    st.subheader("Filtered Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Events",
            len(filtered_df),
        )

    with col2:
        st.metric(
            "High Risk Events",
            high_risk_count,
        )

    with col3:
        st.metric(
            "Medium Risk Events",
            medium_risk_count,
        )

    if high_risk_count > 0:
        st.error(
            f"⚠️ HIGH RISK WARNING: {high_risk_count:,} "
            "high-risk disaster events detected."
        )
    else:
        st.success(
            "✅ No high-risk disaster events detected."
        )

    if medium_risk_count > 0:
        st.warning(
            f"⚠️ {medium_risk_count:,} medium-risk "
            "disaster events detected."
        )

    if low_risk_count > 0:
        st.info(
            f"ℹ️ {low_risk_count:,} low-risk "
            "disaster events detected."
        )

    st.subheader("Risk Distribution")

    fig = create_risk_distribution_chart(filtered_df)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Global Disaster Map")

    map_fig = create_disaster_map(filtered_df)

    st.plotly_chart(
        map_fig,
        use_container_width=True,
    )

    st.subheader("Disaster Events")

    st.dataframe(
        filtered_df,
        use_container_width=True,
    )

except FileNotFoundError as error:
    st.error(str(error))