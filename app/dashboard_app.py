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
from src.visualization import create_risk_distribution_chart


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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Events", len(df))

    with col2:
        st.metric(
            "High Risk Events",
            int((df["risk_level"] == "HIGH").sum()),
        )

    with col3:
        st.metric(
            "Medium Risk Events",
            int((df["risk_level"] == "MEDIUM").sum()),
        )

    st.subheader("Risk Distribution")

    fig = create_risk_distribution_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Disaster Events")

    st.dataframe(
        df,
        use_container_width=True,
    )

except FileNotFoundError as error:
    st.error(str(error))