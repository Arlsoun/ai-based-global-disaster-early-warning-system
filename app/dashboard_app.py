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
# DARK APPLICATION STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL BACKGROUND
       ======================================================== */

    html,
    body,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    .stApp {
        background-color: #07111F !important;
        color: #F8FAFC !important;
    }

    [data-testid="stHeader"] {
        background-color: #07111F !important;
    }

    [data-testid="stToolbar"] {
        background-color: #07111F !important;
    }

    /* Main content */
    [data-testid="stMainBlockContainer"] {
        background-color: #07111F !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0F172A 0%,
            #172554 100%
        ) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] input {
        background-color: #0D1B2E !important;
        color: #F8FAFC !important;
        caret-color: #38BDF8 !important;
    }


    /* ========================================================
       SELECTBOX AND MULTISELECT
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #0D1B2E !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }

    div[data-baseweb="select"] input {
        background-color: transparent !important;
        color: #F8FAFC !important;
        caret-color: #38BDF8 !important;
    }

    div[data-baseweb="select"] input::placeholder {
        color: #94A3B8 !important;
    }

    div[data-baseweb="select"] [data-baseweb="tag"] {
        background-color: #1E3A5F !important;
        color: #F8FAFC !important;
    }

    div[data-baseweb="select"] [data-baseweb="tag"] span {
        color: #F8FAFC !important;
    }

    div[data-baseweb="select"] svg {
        fill: #CBD5E1 !important;
    }

    /* Dropdown menu */

    [data-baseweb="popover"] {
        background-color: #0D1B2E !important;
    }

    [data-baseweb="popover"] > div {
        background-color: #0D1B2E !important;
    }

    [data-baseweb="menu"] {
        background-color: #0D1B2E !important;
        color: #F8FAFC !important;
    }

    li[role="option"] {
        background-color: #0D1B2E !important;
        color: #F8FAFC !important;
    }

    li[role="option"]:hover {
        background-color: #1E3A5F !important;
        color: #FFFFFF !important;
    }

    li[role="option"][aria-selected="true"] {
        background-color: #1E3A5F !important;
        color: #FFFFFF !important;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .main-title {
        color: #F8FAFC !important;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #CBD5E1 !important;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        color: #F8FAFC !important;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1.4rem;
        margin-bottom: 0.9rem;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="metric-container"] {
        background-color: #0D1B2E !important;
        border: 1px solid #334155 !important;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="metric-container"] label {
        color: #CBD5E1 !important;
        font-weight: 650;
    }

    div[data-testid="metric-container"]
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 800;
    }


    /* ========================================================
       RISK STATUS
       ======================================================== */

    .risk-high {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #3B1115 !important;
        border-left: 6px solid #EF4444;
        color: #FCA5A5 !important;
        margin-bottom: 0.8rem;
    }

    .risk-medium {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #3A1F0B !important;
        border-left: 6px solid #F97316;
        color: #FDBA74 !important;
        margin-bottom: 0.8rem;
    }

    .risk-low {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #0B2A1A !important;
        border-left: 6px solid #22C55E;
        color: #86EFAC !important;
        margin-bottom: 0.8rem;
    }

    .info-box {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #0B2340 !important;
        border-left: 6px solid #38BDF8;
        color: #BAE6FD !important;
        margin-bottom: 0.8rem;
    }


    /* ========================================================
       MODEL EVALUATION
       ======================================================== */

    .evaluation-card {
        background-color: #0D1B2E !important;
        border: 1px solid #334155 !important;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25);
    }

    .evaluation-title {
        color: #CBD5E1 !important;
        font-size: 0.9rem;
        font-weight: 650;
    }

    .evaluation-value {
        color: #38BDF8 !important;
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 6px;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        background-color: #0D1B2E !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        overflow: hidden;
    }


    /* ========================================================
       PLOTLY
       ======================================================== */

    div[data-testid="stPlotlyChart"] {
        background-color: #0D1B2E !important;
        border: 1px solid #334155 !important;
        border-radius: 14px;
        padding: 8px;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        font-weight: 650;
    }

    .stButton > button:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }


    /* ========================================================
       TEXT
       ======================================================== */

    .stMarkdown,
    .stCaption,
    p,
    label {
        color: #F8FAFC !important;
    }

    input,
    textarea {
        background-color: #0D1B2E !important;
        color: #F8FAFC !important;
        border-color: #475569 !important;
    }


    /* ========================================================
       SLIDER
       ======================================================== */

    [data-testid="stSlider"] {
        color: #F8FAFC !important;
    }

    [data-testid="stSlider"] label {
        color: #E2E8F0 !important;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: #334155 !important;
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

    st.success("Disaster data loaded successfully.")


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


    # ========================================================
    # FILTERED RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Filtered Results</div>',
        unsafe_allow_html=True,
    )

    total_events = len(filtered_df)

    high_risk = int(
        (filtered_df["risk_level"] == "HIGH").sum()
    )

    medium_risk = int(
        (filtered_df["risk_level"] == "MEDIUM").sum()
    )

    low_risk = int(
        (filtered_df["risk_level"] == "LOW").sum()
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

    fig = create_risk_distribution_chart(filtered_df)

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

    map_fig = create_disaster_map(filtered_df)

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

    column_config = {
        "id": st.column_config.TextColumn(
            "ID",
            width="medium",
        ),
        "country": st.column_config.TextColumn(
            "Country",
            width="medium",
        ),
        "year": st.column_config.NumberColumn(
            "Year",
            width="small",
        ),
        "disastertype": st.column_config.TextColumn(
            "Disaster Type",
            width="medium",
        ),
        "latitude": st.column_config.NumberColumn(
            "Latitude",
            format="%.4f",
            width="medium",
        ),
        "longitude": st.column_config.NumberColumn(
            "Longitude",
            format="%.4f",
            width="medium",
        ),
        "risk_score": st.column_config.NumberColumn(
            "Risk Score",
            format="%.4f",
            width="medium",
        ),
        "predicted_risk_score": st.column_config.NumberColumn(
            "Predicted Risk Score",
            format="%.4f",
            width="large",
        ),
        "risk_level": st.column_config.TextColumn(
            "Risk Level",
            width="large",
        ),
    }

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        height=600,
    )


except FileNotFoundError as error:
    st.error(str(error))

except Exception as error:
    st.error(
        f"An unexpected error occurred: {error}"
    )