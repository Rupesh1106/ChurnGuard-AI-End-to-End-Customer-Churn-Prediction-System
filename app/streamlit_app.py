"""
ChurnGuard AI — Streamlit Web Application
==========================================
Real-time customer churn prediction with interactive UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ────────────────────────────────────────────
# Page Configuration
# ────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────
# Header
# ────────────────────────────────────────────
st.markdown('<h1 class="main-header">🛡️ ChurnGuard AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #888;">'
    'Real-Time Customer Churn Prediction & Analysis</p>',
    unsafe_allow_html=True,
)
st.divider()

# ────────────────────────────────────────────
# Sidebar — Customer Input
# ────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Customer Information")
    st.markdown("---")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown("---")
    st.subheader("📞 Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    st.markdown("---")
    st.subheader("💳 Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=float(tenure * monthly_charges),
    )

    predict_button = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

# ────────────────────────────────────────────
# Main Content — Tabs
# ────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])

with tab1:
    if predict_button:
        # Simulated prediction (replace with model loading in production)
        np.random.seed(hash(f"{tenure}{monthly_charges}{contract}") % 2**32)

        # Heuristic-based demo prediction
        churn_score = 0.0
        if contract == "Month-to-month":
            churn_score += 0.3
        if tenure < 12:
            churn_score += 0.2
        if monthly_charges > 70:
            churn_score += 0.15
        if online_security == "No":
            churn_score += 0.1
        if tech_support == "No":
            churn_score += 0.1
        if payment_method == "Electronic check":
            churn_score += 0.1

        churn_prob = min(churn_score + np.random.uniform(-0.05, 0.05), 0.99)
        churn_prob = max(churn_prob, 0.01)

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        with col2:
            risk = "🔴 High" if churn_prob > 0.6 else "🟡 Medium" if churn_prob > 0.3 else "🟢 Low"
            st.metric("Risk Level", risk)
        with col3:
            st.metric("Confidence", f"{max(churn_prob, 1 - churn_prob):.1%}")

        st.divider()

        # Churn probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            title={"text": "Churn Risk Score", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#764ba2"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader("💡 Retention Recommendations")
        if churn_prob > 0.6:
            st.error("⚠️ **High Risk Customer** — Immediate action recommended!")
            st.markdown("""
            - 🎁 Offer a personalized discount or loyalty reward
            - 📞 Schedule a proactive customer success call
            - 📋 Propose a contract upgrade with added benefits
            - 🛠️ Review and resolve any open support tickets
            """)
        elif churn_prob > 0.3:
            st.warning("⚡ **Medium Risk** — Monitor closely.")
            st.markdown("""
            - 📧 Send targeted engagement emails
            - 🎯 Offer relevant service upgrades
            - 📊 Track usage patterns for early warning signs
            """)
        else:
            st.success("✅ **Low Risk** — Customer appears satisfied.")
            st.markdown("""
            - 🌟 Continue delivering excellent service
            - 💬 Request feedback and reviews
            - 🎉 Include in loyalty/referral programs
            """)
    else:
        st.info("👈 Fill in customer details in the sidebar and click **Predict Churn** to get started.")

with tab2:
    st.subheader("📊 Churn Analytics Dashboard")
    st.markdown("Upload your customer dataset to view analytics.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Churn" in df.columns:
                fig = px.pie(
                    df, names="Churn", title="Churn Distribution",
                    color_discrete_sequence=["#667eea", "#f093fb"],
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "MonthlyCharges" in df.columns:
                fig = px.histogram(
                    df, x="MonthlyCharges", title="Monthly Charges Distribution",
                    color_discrete_sequence=["#667eea"],
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No data uploaded yet.")

with tab3:
    st.subheader("ℹ️ About ChurnGuard AI")
    st.markdown("""
    **ChurnGuard AI** is an end-to-end customer churn prediction system built using:

    - 🐍 **Python** for data processing and ML
    - 🤖 **XGBoost & Random Forest** for high-accuracy churn predictions
    - 🔍 **SHAP** for model explainability
    - 📊 **Power BI** for interactive business dashboards
    - 🌐 **Streamlit** for real-time web-based predictions

    ---

    **Developed by [Rupesh](https://github.com/Rupesh1106)** | MIT License
    """)
