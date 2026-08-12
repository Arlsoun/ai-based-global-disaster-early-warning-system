import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_disaster_data
from src.preprocess import preprocess_disaster_data
from src.disaster_model import calculate_risk_score
from src.ml_model import (
    train_risk_model,
    predict_risk,
    evaluate_risk_model,
)
from src.early_warning import assign_risk_level
from src.visualization import (
    create_risk_distribution_chart,
    create_disaster_map,
)


st.set_page_config(
    page_title="Global Disaster Early-Warning System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM APPLICATION STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APPLICATION
       ======================================================== */

    .stApp {
        background-color: #F8FAFC;
    }

    .main {
        background-color: #F8FAFC;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0F172A 0%,
            #172554 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border: 1px solid #475569 !important;
        border-radius: 8px;
    }

    section[data-testid="stSidebar"] input {
        color: #FFFFFF !important;
    }

    /* ========================================================
       HEADER
       ======================================================== */

    .main-title {
        color: #0F172A;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        color: #1E293B;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1.4rem;
        margin-bottom: 0.9rem;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.07);
    }

    div[data-testid="metric-container"] label {
        color: #64748B !important;
        font-weight: 650;
    }

    div[data-testid="metric-container"]
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800;
    }

    /* ========================================================
       RISK STATUS
       ======================================================== */

    .risk-high {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #FEF2F2;
        border-left: 6px solid #DC2626;
        color: #991B1B;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 7px rgba(220, 38, 38, 0.08);
    }

    .risk-medium {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #FFF7ED;
        border-left: 6px solid #EA580C;
        color: #9A3412;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 7px rgba(234, 88, 12, 0.08);
    }

    .risk-low {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #F0FDF4;
        border-left: 6px solid #16A34A;
        color: #166534;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 7px rgba(22, 163, 74, 0.08);
    }

    .info-box {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #EFF6FF;
        border-left: 6px solid #2563EB;
        color: #1E40AF;
        margin-bottom: 0.8rem;
    }

    /* ========================================================
       MODEL EVALUATION CARDS
       ======================================================== */

    .evaluation-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.07);
    }

    .evaluation-title {
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 650;
    }

    .evaluation-value {
        color: #2563EB;
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 6px;
    }

    /* ========================================================
       STREAMLIT ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ========================================================
       SELECT BOXES
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
    }

    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: #2563EB;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 650;
    }

    .stButton > button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
    }

    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #E2E8F0;
    }

    /* ========================================================
       PLOTLY CONTAINERS
       ======================================================== */

    div[data-testid="stPlotlyChart"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 8px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🌍 AI-Based Global Disaster Early-Warning System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Disaster-event analysis and early-warning risk assessment
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    # ========================================================
    # DATA PROCESSING
    # ========================================================

    df = load_disaster_data()
    df = preprocess_disaster_data(df)
    df = calculate_risk_score(df)

    model = train_risk_model(df)
    evaluation = evaluate_risk_model(model, df)

    df = predict_risk(model, df)
    df = assign_risk_level(df)

    st.success(
        "Disaster data loaded successfully."
    )


    # ========================================================
    # SIDEBAR FILTERS
    # ========================================================

    st.sidebar.header("🔎 Dashboard Filters")

    st.sidebar.caption(
        "Use the filters below to explore disaster risk."
    )

    countries = sorted(
        df["country"]
        .dropna()
        .unique()
        .tolist()
    )

    disaster_types = sorted(
        df["disastertype"]
        .dropna()
        .unique()
        .tolist()
    )

    risk_levels = [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

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


    # ========================================================
    # FILTER DATA
    # ========================================================

    filtered_df = df.copy()

    if selected_countries:
        filtered_df = filtered_df[
            filtered_df["country"].isin(
                selected_countries
            )
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


    # ========================================================
    # FILTERED RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Filtered Results</div>',
        unsafe_allow_html=True,
    )

    total_events = len(filtered_df)

    high_risk = int(
        (
            filtered_df["risk_level"] == "HIGH"
        ).sum()
    )

    medium_risk = int(
        (
            filtered_df["risk_level"] == "MEDIUM"
        ).sum()
    )

    low_risk = int(
        (
            filtered_df["risk_level"] == "LOW"
        ).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Events",
            f"{total_events:,}",
        )

    with col2:
        st.metric(
            "🔴 High Risk",
            f"{high_risk:,}",
        )

    with col3:
        st.metric(
            "🟠 Medium Risk",
            f"{medium_risk:,}",
        )

    with col4:
        st.metric(
            "🟢 Low Risk",
            f"{low_risk:,}",
        )


    # ========================================================
    # WARNING STATUS
    # ========================================================

    st.markdown(
        '<div class="section-title">⚠️ Warning Status</div>',
        unsafe_allow_html=True,
    )

    if high_risk > 0:
        st.markdown(
            f"""
            <div class="risk-high">
                🔴 <strong>HIGH RISK</strong><br>
                {high_risk:,} high-risk disaster events detected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if medium_risk > 0:
        st.markdown(
            f"""
            <div class="risk-medium">
                🟠 <strong>MEDIUM RISK</strong><br>
                {medium_risk:,} medium-risk disaster events detected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if low_risk > 0:
        st.markdown(
            f"""
            <div class="risk-low">
                🟢 <strong>LOW RISK</strong><br>
                {low_risk:,} low-risk disaster events detected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if total_events == 0:
        st.markdown(
            """
            <div class="info-box">
                ℹ️ No disaster events match the selected filters.
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🤖 Model Evaluation</div>',
        unsafe_allow_html=True,
    )

    eval_col1, eval_col2, eval_col3 = st.columns(3)

    with eval_col1:
        st.markdown(
            f"""
            <div class="evaluation-card">
                <div class="evaluation-title">
                    Mean Absolute Error
                </div>
                <div class="evaluation-value">
                    {evaluation["mae"]:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with eval_col2:
        st.markdown(
            f"""
            <div class="evaluation-card">
                <div class="evaluation-title">
                    Root Mean Squared Error
                </div>
                <div class="evaluation-value">
                    {evaluation["rmse"]:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with eval_col3:
        st.markdown(
            f"""
            <div class="evaluation-card">
                <div class="evaluation-title">
                    R² Score
                </div>
                <div class="evaluation-value">
                    {evaluation["r2"]:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Evaluation metrics are calculated using the full disaster dataset."
    )


    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Risk Distribution</div>',
        unsafe_allow_html=True,
    )

    fig = create_risk_distribution_chart(
        filtered_df
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # GLOBAL DISASTER MAP
    # ========================================================

    st.markdown(
        '<div class="section-title">🌎 Global Disaster Map</div>',
        unsafe_allow_html=True,
    )

    map_fig = create_disaster_map(
        filtered_df
    )

    st.plotly_chart(
        map_fig,
        use_container_width=True,
    )


    # ========================================================
    # DISASTER EVENTS
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Disaster Events</div>',
        unsafe_allow_html=True,
    )

    display_columns = [
        column
        for column in [
            "id",
            "country",
            "year",
            "disastertype",
            "latitude",
            "longitude",
            "risk_score",
            "predicted_risk_score",
            "risk_level",
        ]
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )


except FileNotFoundError as error:
    st.error(str(error))