"""
ui.py — Malaria Severity AI Dashboard
------------------------------------------------
Production-ready Streamlit frontend for a safety-aware explainable
hybrid clinical decision support prototype.

Purpose:
This frontend connects to the FastAPI backend and presents a polished
clinical decision support dashboard for malaria severity prediction.

Important clinical note:
This dashboard is for research, education, and prototype demonstration.
It is not a substitute for professional diagnosis, medical examination,
or laboratory confirmation.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ------------------------------------------------
# GLOBAL CONFIGURATION
# ------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"
CURRENT_YEAR = dt.datetime.now().year

st.set_page_config(
    page_title="Malaria Severity AI Dashboard",
    page_icon="🧬",
    layout="wide",
)


# ------------------------------------------------
# CUSTOM CSS FOR EXECUTIVE DASHBOARD LOOK
# ------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7fbff 0%, #eef5ff 100%);
    }

    .hero-card {
        padding: 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, #061b2f, #0b4778);
        color: white;
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 24px;
        font-weight: 700;
        margin-top: 8px;
    }

    .hero-text {
        font-size: 16px;
        margin-top: 18px;
    }

    .metric-card {
        padding: 22px;
        border-radius: 20px;
        background: linear-gradient(135deg, #061b2f, #0b2f4a);
        color: white;
        text-align: center;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
        min-height: 136px;
    }

    .metric-title {
        font-size: 15px;
        opacity: 0.82;
        font-weight: 700;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 900;
        margin-top: 15px;
    }

    .risk-low {
        background: #d1fae5;
        color: #065f46;
        padding: 18px;
        border-radius: 16px;
        font-weight: 800;
        border-left: 7px solid #10b981;
    }

    .risk-medium {
        background: #fef3c7;
        color: #92400e;
        padding: 18px;
        border-radius: 16px;
        font-weight: 800;
        border-left: 7px solid #f59e0b;
    }

    .risk-high {
        background: #fee2e2;
        color: #991b1b;
        padding: 18px;
        border-radius: 16px;
        font-weight: 800;
        border-left: 7px solid #ef4444;
    }

    .recommend-card {
        padding: 16px;
        border-radius: 15px;
        background: #ffffff;
        border: 1px solid #dbeafe;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }

    .critical-card {
        padding: 14px;
        border-radius: 14px;
        background: #fff7ed;
        border-left: 7px solid #f97316;
        color: #9a3412;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .safe-card {
        padding: 14px;
        border-radius: 14px;
        background: #dcfce7;
        border-left: 7px solid #22c55e;
        color: #166534;
        font-weight: 800;
    }

    .disclaimer-box {
        background: #fff7ed;
        color: #9a3412;
        border-left: 7px solid #f97316;
        padding: 16px;
        border-radius: 16px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    .footer-card {
        margin-top: 36px;
        padding: 22px;
        border-radius: 18px;
        background: linear-gradient(135deg, #061b2f, #0b2f4a);
        color: white;
        text-align: center;
        box-shadow: 0 8px 22px rgba(0,0,0,0.16);
        font-size: 15px;
    }

    .top-link {
        display: inline-block;
        margin-top: 14px;
        padding: 10px 18px;
        background: #e0f2fe;
        color: #075985 !important;
        text-decoration: none !important;
        border-radius: 999px;
        font-weight: 900;
        border: 1px solid #7dd3fc;
    }

    .top-link:hover {
        background: #bae6fd;
        color: #0c4a6e !important;
    }

    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------
# TOP ANCHOR FOR RETURN-TO-TOP LINK
# ------------------------------------------------
st.markdown('<div id="top"></div>', unsafe_allow_html=True)


# ------------------------------------------------
# FEATURE GROUPS
# ------------------------------------------------
GENERAL_SYMPTOMS = [
    "fever", "cold", "rigor", "fatigue", "headache",
    "bitter_tongue", "vomiting", "diarrhea",
]

SEVERE_INDICATORS = [
    "convulsion", "anemia", "jaundice", "coca_cola_urine",
    "hypoglycemia", "prostration", "hyperpyrexia",
]

ALL_SYMPTOMS = GENERAL_SYMPTOMS + SEVERE_INDICATORS


# ------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------
def initialize_session_state() -> None:
    """Initializes all persistent Streamlit session values."""

    defaults = {
        "result": None,
        "payload": None,
        "age": 25,
        "sex_label": "Female",
        "select_all_symptoms": False,
        "select_all_general": False,
        "select_all_severe": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for symptom in ALL_SYMPTOMS:
        if symptom not in st.session_state:
            st.session_state[symptom] = False


initialize_session_state()


# ------------------------------------------------
# STATE CONTROL CALLBACKS
# ------------------------------------------------
def apply_global_select_all() -> None:
    """Applies the global Select All checkbox to every symptom group."""

    selected = st.session_state.select_all_symptoms
    st.session_state.select_all_general = selected
    st.session_state.select_all_severe = selected

    for symptom in ALL_SYMPTOMS:
        st.session_state[symptom] = selected


def apply_general_select_all() -> None:
    """Applies Select All only to the general symptom group."""

    selected = st.session_state.select_all_general

    for symptom in GENERAL_SYMPTOMS:
        st.session_state[symptom] = selected


def apply_severe_select_all() -> None:
    """Applies Select All only to the severe malaria indicator group."""

    selected = st.session_state.select_all_severe

    for symptom in SEVERE_INDICATORS:
        st.session_state[symptom] = selected


def reset_dashboard() -> None:
    """Safely resets dashboard values by rebuilding widget state."""

    keys_to_clear = [
        "result", "payload", "age", "sex_label", "select_all_symptoms",
        "select_all_general", "select_all_severe",
    ]
    keys_to_clear.extend(ALL_SYMPTOMS)

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    initialize_session_state()


# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def yes_no(value: bool) -> int:
    """Converts checkbox boolean values into model-ready binary values."""
    return 1 if value else 0


def format_feature_name(name: str) -> str:
    """Converts snake_case feature names into clinician-readable labels."""
    return name.replace("_", " ").title()


def format_decision_source(source: str) -> str:
    """Converts backend decision source into a readable dashboard label."""

    mapping = {
        "clinical_override": "🛡️ Safety Escalation Active",
        "clinical_baseline_guardrail": "🛡️ Baseline Clinical Guardrail",
        "model_probability": "🧠 Model Probability",
        "symptom_burden_guardrail": "👁️ Symptom Burden Guardrail",
    }

    return mapping.get(source, source.replace("_", " ").title())


def call_api(payload: dict) -> tuple[dict | None, str | None]:
    """Sends patient data to FastAPI backend and returns prediction result."""

    try:
        response = requests.post(API_URL, json=payload, timeout=15)

        if response.status_code == 200:
            return response.json(), None

        return None, f"API Error {response.status_code}: {response.text}"

    except requests.exceptions.ConnectionError:
        return None, "Could not connect to FastAPI backend. Run: uvicorn src.app:app --reload"

    except Exception as error:
        return None, str(error)


def get_risk_icon(risk: str) -> str:
    """Returns visual icon based on risk level."""

    if risk == "HIGH":
        return "🔴"

    if risk == "MODERATE":
        return "🟠"

    return "🟢"


def clinical_recommendations(result: dict) -> list[tuple[str, str]]:
    """Generates clinician-readable recommendations based on hybrid risk logic."""

    risk = result.get("severity_risk", "UNKNOWN")
    active_symptom_count = result.get("active_symptom_count", 0)
    critical_flags = result.get("critical_flags", 0)

    if risk == "HIGH" or critical_flags > 0:
        return [
            ("🚨 Immediate Action", "Immediate physician review is recommended."),
            ("🏥 Severe Malaria Protocol", "Consider severe malaria management protocol."),
            ("🧠 Neurological Monitoring", "Monitor neurological signs and blood glucose closely."),
            ("🧪 Laboratory Confirmation", "Urgent laboratory confirmation and escalation may be required."),
            ("⚖️ Clinical Caution", "Use AI output as support, not as final diagnosis."),
        ]

    if risk == "MODERATE" or active_symptom_count >= 4:
        return [
            ("👁️ Close Observation", "Monitor patient closely because multiple malaria-consistent symptoms are present."),
            ("🧪 Confirmatory Testing", "Request malaria test confirmation and relevant clinical/laboratory assessment."),
            ("💧 Hydration / General Status", "Monitor hydration, weakness, vomiting, diarrhea, and functional status."),
            ("🔁 Reassessment", "Reassess promptly if symptoms persist, worsen, or new severe indicators appear."),
            ("⚖️ Clinical Caution", "Use AI output as support, not as final diagnosis."),
        ]

    return [
        ("✅ Routine Monitoring", "Routine monitoring may be sufficient."),
        ("📅 Follow-up", "Advise follow-up if symptoms persist or worsen."),
        ("🧪 Confirmation", "Confirm with standard clinical evaluation."),
        ("⚖️ Clinical Caution", "Use AI output as support, not as final diagnosis."),
    ]


def render_risk_box(result: dict) -> None:
    """Displays color-coded risk alert panel."""

    risk = result.get("severity_risk", "UNKNOWN")

    if risk == "HIGH":
        st.markdown(
            '<div class="risk-high">🔴 HIGH RISK — Immediate attention required</div>',
            unsafe_allow_html=True,
        )

    elif risk == "MODERATE":
        st.markdown(
            '<div class="risk-medium">🟠 MODERATE RISK — Close observation recommended</div>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            '<div class="risk-low">🟢 LOW RISK — Stable condition</div>',
            unsafe_allow_html=True,
        )


def render_metric_card(title: str, value: str) -> None:
    """Displays a reusable executive metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_critical_symptom_panel(result: dict) -> None:
    """Displays active clinical override indicators as warning cards."""

    active_critical = result.get("active_critical_symptoms", [])

    if active_critical:
        for symptom in active_critical:
            st.markdown(
                f"""
                <div class="critical-card">
                    ⚠ {format_feature_name(symptom)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="safe-card">✅ No critical clinical override indicators were detected.</div>',
            unsafe_allow_html=True,
        )


def render_recommendation_cards(result: dict) -> None:
    """Displays clinical recommendations in clean executive cards."""

    for title, message in clinical_recommendations(result):
        st.markdown(
            f"""
            <div class="recommend-card">
                <strong>{title}</strong><br>
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_warning(result: dict) -> None:
    """Shows model transparency warning returned by the backend."""

    warning = result.get("model_warning", "")

    if warning:
        st.warning(warning)


def render_explainability_chart(df: pd.DataFrame, chart_type: str, shap_title: str) -> None:
    """Renders selectable SHAP visualization."""

    if df.empty:
        st.info("No SHAP explanation data available.")
        return

    df = df.copy()
    df["abs_impact"] = df["impact"].abs()
    df["feature_label"] = df["feature"].apply(format_feature_name)
    df = df.sort_values("abs_impact", ascending=False)

    if chart_type == "Horizontal Bar Chart":
        fig = px.bar(
            df,
            x="impact",
            y="feature_label",
            orientation="h",
            title=shap_title,
            labels={"impact": "SHAP Impact", "feature_label": "Clinical Feature"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Vertical Bar Chart":
        fig = px.bar(
            df,
            x="feature_label",
            y="abs_impact",
            title=f"{shap_title} — Absolute Ranking",
            labels={"abs_impact": "Absolute SHAP Impact", "feature_label": "Clinical Feature"},
        )
        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Clinical Ranking Table":
        table = df[["feature_label", "impact", "direction"]].rename(
            columns={
                "feature_label": "Clinical Feature",
                "impact": "SHAP Contribution",
                "direction": "Effect on Model Severe-Risk Probability",
            }
        )
        table["SHAP Contribution"] = table["SHAP Contribution"].round(4)
        st.dataframe(table, width="stretch")

    elif chart_type == "Pie Chart":
        fig = px.pie(df, names="feature_label", values="abs_impact", title="Relative SHAP Contribution Share")
        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Radar Chart":
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=df["abs_impact"],
                theta=df["feature_label"],
                fill="toself",
                name="SHAP Impact",
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=False,
            title="Radar View of Feature Contributions",
        )
        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Waterfall Style":
        fig = go.Figure(
            go.Waterfall(
                name="SHAP",
                orientation="v",
                measure=["relative"] * len(df),
                x=df["feature_label"],
                y=df["impact"],
                text=df["impact"].round(4),
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Waterfall-style SHAP Contribution",
            xaxis_title="Clinical Feature",
            yaxis_title="Impact on Model Severe-Risk Probability",
        )
        st.plotly_chart(fig, width="stretch")


def render_footer() -> None:
    """Displays copyright footer and return-to-top link."""

    st.markdown(
        f"""
        <div class="footer-card">
            © {CURRENT_YEAR} ADA Global Academy Data Science Scholar.
            All rights reserved. Malaria Severity AI Dashboard.
            <br>
            <span style="opacity:0.85;">
                Research prototype for explainable hybrid clinical decision support.
            </span>
            <br>
            <a href="#top" class="top-link">⬆ Return to Top</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------
# HEADER / HERO SECTION
# ------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🧬 Malaria Severity AI Dashboard</div>
        <div class="hero-subtitle">Safety-Aware Explainable Hybrid Clinical Decision Support System</div>
        <div class="hero-text">
            Powered by RandomForest, Clinical Safety Guardrails, SHAP Explainability, and human-readable audit evidence.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ Research Prototype Disclaimer: This dashboard is for academic, research, and demonstration use only.
        It is not a medical device and must not be used as a final diagnosis. Always confirm with qualified
        clinical evaluation and laboratory testing.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------
# MULTI-TAB DASHBOARD LAYOUT
# ------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🩺 Clinical Prediction",
        "📊 Explainability Analytics",
        "📚 Research Dashboard",
        "⚙️ Model/API Information",
    ]
)


# ------------------------------------------------
# TAB 1 — CLINICAL PREDICTION
# ------------------------------------------------
with tab1:
    with st.container(border=True):
        st.subheader("👤 Patient Clinical Information")

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.slider("Age", min_value=1, max_value=100, key="age")
            st.radio("Sex", ["Female", "Male"], horizontal=True, key="sex_label")

        with col_b:
            st.info(
                "Use the grouped controls below to simulate patient symptom profiles. "
                "This dashboard supports clinical AI demonstration, research testing, "
                "explainability review, and safety-aware decision support design."
            )

    with st.container(border=True):
        st.subheader("🧾 Symptom Selection")

        control_col1, control_col2 = st.columns([1, 1])

        with control_col1:
            st.checkbox("✅ Select ALL Symptoms", key="select_all_symptoms", on_change=apply_global_select_all)

        with control_col2:
            if st.button("🔄 Reset Patient Form", width="stretch"):
                reset_dashboard()
                st.rerun()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌡️ General Symptoms")
            st.checkbox("Select All General Symptoms", key="select_all_general", on_change=apply_general_select_all)

            for symptom in GENERAL_SYMPTOMS:
                st.checkbox(format_feature_name(symptom), key=symptom)

        with col2:
            st.markdown("### 🚨 Severe Malaria Indicators")
            st.checkbox("Select All Severe Indicators", key="select_all_severe", on_change=apply_severe_select_all)

            for symptom in SEVERE_INDICATORS:
                st.checkbox(format_feature_name(symptom), key=symptom)

        st.divider()

        if st.button("🔍 Analyze Patient", width="stretch"):
            sex_value = 1 if st.session_state.sex_label == "Male" else 0

            payload = {"age": st.session_state.age, "sex": sex_value}

            for symptom in ALL_SYMPTOMS:
                payload[symptom] = yes_no(st.session_state[symptom])

            result, error = call_api(payload)

            if error:
                st.error(error)
            else:
                st.session_state.result = result
                st.session_state.payload = payload

    result = st.session_state.result

    if result:
        with st.container(border=True):
            st.subheader("📋 Diagnosis Summary")

            c1, c2, c3, c4, c5, c6 = st.columns(6)

            final_label = result.get("label", "Unknown")
            model_label = result.get("model_label", "Unknown")
            probability = result.get("probability_severe", 0) * 100
            certainty = result.get("prediction_certainty", 0) * 100
            risk = result.get("severity_risk", "UNKNOWN")
            risk_icon = get_risk_icon(risk)
            decision_source = format_decision_source(result.get("risk_basis", ""))

            with c1:
                render_metric_card("Final Clinical Decision", f"🩺 {final_label}")

            with c2:
                render_metric_card("Model Prediction", f"🤖 {model_label}")

            with c3:
                render_metric_card("Risk Level", f"{risk_icon} {risk}")

            with c4:
                render_metric_card("Probability", f"{probability:.2f}%")

            with c5:
                render_metric_card("Decision Source", decision_source)

            with c6:
                render_metric_card("Prediction Certainty", f"{certainty:.1f}%")

            hybrid_reasoning = result.get("hybrid_reasoning", "")
            if hybrid_reasoning:
                st.info(f"🧠 Hybrid Decision Reasoning: {hybrid_reasoning}")

            st.markdown("### 🚦 Risk Indicator")
            render_risk_box(result)
            st.progress(min(result.get("probability_severe", 0), 1.0))

        with st.container(border=True):
            st.subheader("🚨 Critical Symptoms Panel")
            render_critical_symptom_panel(result)

        with st.container(border=True):
            st.subheader("💡 Clinical Recommendation")
            render_recommendation_cards(result)


# ------------------------------------------------
# TAB 2 — EXPLAINABILITY ANALYTICS
# ------------------------------------------------
with tab2:
    with st.container(border=True):
        st.subheader("📊 SHAP Explainability Center")

        result = st.session_state.result

        if not result:
            st.info("Run a prediction first from the Clinical Prediction tab.")

        else:
            top_contributors = result.get("top_contributors", [])
            df_shap = pd.DataFrame(top_contributors)
            risk_basis = result.get("risk_basis", "")

            if risk_basis == "clinical_override":
                shap_title = "SHAP Feature Impact on Model Probability"
            else:
                shap_title = "SHAP Feature Impact on Severe Malaria Prediction"

            chart_type = st.selectbox(
                "Select Explainability Visualization",
                [
                    "Horizontal Bar Chart",
                    "Vertical Bar Chart",
                    "Clinical Ranking Table",
                    "Pie Chart",
                    "Radar Chart",
                    "Waterfall Style",
                ],
            )

            render_explainability_chart(df_shap, chart_type, shap_title)

            st.info(
                "Note: SHAP explains machine-learning probability patterns only. "
                "Final clinical decisions may additionally include rule-based "
                "safety overrides for high-risk symptoms."
            )

            render_model_warning(result)

            st.markdown("### 🧠 AI Explanation Summary")
            st.info(result.get("explanation_summary", "No explanation summary available."))

            st.markdown("### 🏥 Clinical Safety Summary")
            st.warning(result.get("clinical_summary", "No clinical summary available."))

            with st.expander("View full SHAP values"):
                df_all = pd.DataFrame(result.get("all_shap_values", []))

                if not df_all.empty:
                    # ---------------------------------------------------
                    # BUILD CLINICALLY HONEST SHAP TABLE
                    # ---------------------------------------------------
                    df_all["Clinical Feature"] = df_all["feature"].str.replace("_", " ").str.title()

                    # ---------------------------------------------------
                    # DETERMINE CLINICAL STATUS
                    # ---------------------------------------------------
                    def determine_status(feature_name: str) -> str:
                        """Labels each feature as Present, Absent, or Baseline Variable."""

                        feature_key = feature_name.lower().replace(" ", "_")

                        if feature_key in ["age", "sex"]:
                            return "Baseline Variable"

                        payload = st.session_state.payload or {}
                        return "Present" if payload.get(feature_key, 0) == 1 else "Absent"

                    df_all["Clinical Status"] = df_all["Clinical Feature"].apply(determine_status)

                    # ---------------------------------------------------
                    # RENAME AND CLEAN CLINICAL COLUMNS
                    # ---------------------------------------------------
                    df_all = df_all.rename(
                        columns={
                            "impact": "SHAP Contribution",
                            "direction": "Effect on Model Severe-Risk Probability",
                        }
                    )

                    df_all["SHAP Contribution"] = df_all["SHAP Contribution"].round(4)

                    # ---------------------------------------------------
                    # SORT BY STRONGEST ABSOLUTE MODEL IMPACT
                    # ---------------------------------------------------
                    df_all = df_all.sort_values(
                        by="SHAP Contribution",
                        key=lambda col: col.abs(),
                        ascending=False,
                    )

                    st.dataframe(
                        df_all[
                            [
                                "Clinical Feature",
                                "SHAP Contribution",
                                "Effect on Model Severe-Risk Probability",
                                "Clinical Status",
                            ]
                        ],
                        width="stretch",
                    )

                else:
                    st.info("No detailed SHAP values returned.")


# ------------------------------------------------
# TAB 3 — RESEARCH DASHBOARD
# ------------------------------------------------
with tab3:
    with st.container(border=True):
        st.subheader("📚 Research Dashboard")
        st.markdown(
            """
            This prototype has evolved from a simple malaria classifier into a
            **safety-aware hybrid explainable clinical decision support system**.

            The key research contribution is not only prediction, but the design of a
            trustworthy healthcare AI architecture that combines:

            - Machine learning probability estimation
            - Clinical safety override rules
            - Symptom-burden guardrails
            - SHAP-based local explainability
            - Human-readable recommendation support
            - Audit-friendly API response structure
            """
        )

    with st.container(border=True):
        st.subheader("📈 Current Model Performance Snapshot")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            render_metric_card("Accuracy", "63.24%")

        with metric_col2:
            render_metric_card("ROC-AUC", "58.45%")

        with metric_col3:
            render_metric_card("Severe Recall", "26%")

        with metric_col4:
            render_metric_card("Architecture", "Hybrid AI")

        st.warning(
            "Research Note: The model has modest statistical performance. "
            "The hybrid clinical override and symptom-burden guardrail layers "
            "improve safety behaviour for high-risk and gray-zone symptom patterns."
        )

    with st.container(border=True):
        st.subheader("🧪 Latest Prediction Evidence")

        if st.session_state.result:
            st.json(st.session_state.result)
        else:
            st.info("No prediction evidence available yet.")


# ------------------------------------------------
# TAB 4 — MODEL / API INFORMATION
# ------------------------------------------------
with tab4:
    with st.container(border=True):
        st.subheader("⚙️ Backend API Information")

        st.write("Backend API endpoint:")
        st.code(API_URL)

        st.write("Expected backend command:")
        st.code("uvicorn src.app:app --reload")

        st.write("Expected frontend command:")
        st.code("streamlit run src/ui.py")

    with st.container(border=True):
        st.subheader("🧠 Model Architecture")
        st.markdown(
            """
            - **Model:** RandomForestClassifier
            - **Decision Logic:** Model probability + clinical override + symptom-burden guardrail
            - **Explainability:** SHAP TreeExplainer
            - **Backend:** FastAPI
            - **Frontend:** Streamlit
            - **Visualization:** Plotly
            - **Clinical Safety:** Human-readable audit evidence and disclaimer layer
            """
        )

    with st.container(border=True):
        st.subheader("📦 Current API Payload")

        if st.session_state.payload:
            st.json(st.session_state.payload)
        else:
            st.info("No prediction payload yet.")


# ------------------------------------------------
# GLOBAL FOOTER
# ------------------------------------------------
render_footer()
