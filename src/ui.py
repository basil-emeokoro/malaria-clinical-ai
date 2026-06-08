"""
ui.py — Malaria Severity AI Dashboard V4
------------------------------------------------
Research + Executive Streamlit Frontend

Purpose:
This frontend connects to the FastAPI backend and presents a polished
clinical decision support dashboard for malaria severity prediction.

Core Features:
✅ Multi-tab executive dashboard
✅ FastAPI prediction integration
✅ Group-based symptom selection
✅ Select All and Reset controls
✅ SHAP explainability charts
✅ Clinical safety layer visibility
✅ Clean separated frames using Streamlit containers
✅ Research dashboard and model information tab
"""

# ------------------------------------------------
# IMPORTS
# ------------------------------------------------
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ------------------------------------------------
# GLOBAL CONFIGURATION
# ------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"
CURRENT_YEAR = datetime.now().year

st.set_page_config(
    page_title="Malaria Severity AI Dashboard",
    page_icon="🧬",
    layout="wide"
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
        margin-bottom: 25px;
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
        padding: 24px;
        border-radius: 20px;
        background: linear-gradient(135deg, #061b2f, #0b2f4a);
        color: white;
        text-align: center;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
        min-height: 130px;
    }

    .metric-title {
        font-size: 16px;
        opacity: 0.82;
        font-weight: 600;
    }

    .metric-value {
        font-size: 27px;
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
        font-weight: 700;
        margin-bottom: 10px;
    }

    .safe-card {
        padding: 14px;
        border-radius: 14px;
        background: #dcfce7;
        border-left: 7px solid #22c55e;
        color: #166534;
        font-weight: 700;
    }

    .section-note {
        color: #475569;
        font-size: 15px;
    }

    .app-footer {
        margin-top: 32px;
        padding: 22px 18px;
        border-top: 1px solid #cbd5e1;
        color: #334155;
        text-align: center;
        font-size: 14px;
        line-height: 1.6;
    }

    .top-link {
        display: inline-block;
        margin-top: 12px;
        color: #0b4778 !important;
        font-weight: 800;
        text-decoration: none;
    }

    .top-link:hover {
        text-decoration: underline;
    }

    /* ------------------------------------------
    FIX SELECTBOX CURSOR POINTER
    ------------------------------------------- */

    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div id="top"></div>', unsafe_allow_html=True)


# ------------------------------------------------
# FEATURE GROUPS
# ------------------------------------------------
GENERAL_SYMPTOMS = [
    "fever",
    "cold",
    "rigor",
    "fatigue",
    "headache",
    "bitter_tongue",
    "vomiting",
    "diarrhea"
]

SEVERE_INDICATORS = [
    "convulsion",
    "anemia",
    "jaundice",
    "coca_cola_urine",
    "hypoglycemia",
    "prostration",
    "hyperpyrexia"
]

ALL_SYMPTOMS = GENERAL_SYMPTOMS + SEVERE_INDICATORS


# ------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------
def initialize_session_state():
    """
    Initializes all persistent Streamlit session values.

    This prevents checkbox states from behaving unpredictably
    when the app reruns after user interaction.
    """

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
def apply_global_select_all():
    """
    Applies the global Select All checkbox to every symptom group.
    """

    selected = st.session_state.select_all_symptoms

    st.session_state.select_all_general = selected
    st.session_state.select_all_severe = selected

    for symptom in ALL_SYMPTOMS:
        st.session_state[symptom] = selected


def apply_general_select_all():
    """
    Applies Select All only to the general symptom group.
    """

    selected = st.session_state.select_all_general

    for symptom in GENERAL_SYMPTOMS:
        st.session_state[symptom] = selected


def apply_severe_select_all():
    """
    Applies Select All only to the severe malaria indicator group.
    """

    selected = st.session_state.select_all_severe

    for symptom in SEVERE_INDICATORS:
        st.session_state[symptom] = selected


def reset_dashboard():
    """
    Safely resets dashboard values.

    Streamlit widgets cannot be reassigned directly
    after instantiation. Therefore, widget keys are
    removed and rebuilt cleanly.
    """

    keys_to_clear = [
        "result",
        "payload",
        "age",
        "sex_label",
        "select_all_symptoms",
        "select_all_general",
        "select_all_severe"
    ]

    # Add all symptom checkbox keys
    keys_to_clear.extend(ALL_SYMPTOMS)

    # Safely delete existing widget states
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Reinitialize dashboard defaults
    initialize_session_state()


# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def yes_no(value: bool) -> int:
    """
    Converts checkbox boolean values into model-ready binary values.
    """

    return 1 if value else 0


def format_feature_name(name: str) -> str:
    """
    Converts snake_case feature names into clinician-readable labels.
    """

    return name.replace("_", " ").title()


def format_decision_source(source: str) -> str:
    """
    Converts backend decision source into a clinician-safe display label.

    The label intentionally avoids exposing raw backend codes directly
    to end users.
    """

    if source == "clinical_override":
        return "🛡️ Safety Escalation Active"

    if source == "clinical_baseline_guardrail":
        return "🩺 Baseline Clinical Safety Assessment"

    if source == "clinical_safety_confirmed":
        return "Clinical Safety Confirmed"

    if source == "symptom_burden_guardrail":
        return "👁️ Symptom Burden Guardrail"

    if source in ["model_probability", "model_prediction"]:
        return "🧠 Model Probability"

    return source.replace("_", " ").title()


def call_api(payload: dict):
    """
    Sends patient data to FastAPI backend and returns prediction result.
    """

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
    """
    Returns visual icon based on risk level.
    """

    if risk == "HIGH":
        return "🔴"

    if risk == "MODERATE":
        return "🟠"

    return "🟢"


def clinical_recommendations(result: dict):
    """
    Generates clinician-readable recommendations based on model result.
    """

    risk = result.get("severity_risk", "UNKNOWN")

    if risk == "HIGH":
        return [
            ("🚨 Immediate Action", "Immediate physician review is recommended."),
            ("🏥 Severe Malaria Protocol", "Consider severe malaria management protocol."),
            ("🧠 Neurological Monitoring", "Monitor neurological signs and blood glucose closely."),
            ("🧪 Laboratory Confirmation", "Urgent laboratory confirmation and escalation may be required.")
        ]

    if risk == "MODERATE":
        return [
            ("👁️ Close Monitoring", "Monitor patient closely."),
            ("🧪 Confirmatory Testing", "Request clinical and laboratory assessment."),
            ("🔁 Reassessment", "Reassess symptoms if condition worsens."),
            ("📌 Observation", "Consider observation and follow-up testing.")
        ]

    return [
        ("✅ Routine Monitoring", "Routine monitoring may be sufficient."),
        ("📅 Follow-up", "Advise follow-up if symptoms persist or worsen."),
        ("🧪 Confirmation", "Confirm with standard clinical evaluation."),
        ("⚖️ Clinical Caution", "Use AI output as support, not as final diagnosis.")
    ]


def render_risk_box(result: dict):
    """
    Displays color-coded risk alert panel.
    """

    risk = result.get("severity_risk", "UNKNOWN")

    if risk == "HIGH":
        st.markdown(
            '<div class="risk-high">🔴 HIGH RISK — Immediate attention required</div>',
            unsafe_allow_html=True
        )

    elif risk == "MODERATE":
        st.markdown(
            '<div class="risk-medium">🟠 MODERATE RISK — Monitor closely</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="risk-low">🟢 LOW RISK — Stable condition</div>',
            unsafe_allow_html=True
        )


def render_metric_card(title: str, value: str):
    """
    Displays a reusable executive metric card.
    """

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_critical_symptom_panel(result: dict):
    """
    Displays active clinical safety indicators as warning cards.
    """

    active_critical = result.get("active_critical_symptoms", [])

    if active_critical:
        for symptom in active_critical:
            st.markdown(
                f"""
                <div class="critical-card">
                    ⚠ {format_feature_name(symptom)}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="safe-card">✅ No critical clinical safety indicators were detected.</div>',
            unsafe_allow_html=True
        )


def render_recommendation_cards(result: dict):
    """
    Displays clinical recommendations in clean executive cards.
    """

    for title, message in clinical_recommendations(result):
        st.markdown(
            f"""
            <div class="recommend-card">
                <strong>{title}</strong><br>
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_explainability_chart(df: pd.DataFrame, chart_type: str, risk_basis: str = ""):
    """
    Renders selectable SHAP visualization.

    The same SHAP values can be visualized in different formats
    depending on user preference and presentation context.
    """

    if df.empty:
        st.info("No SHAP explanation data available.")
        return

    df = df.copy()
    df["abs_impact"] = df["impact"].abs()
    df["feature_label"] = df["feature"].apply(format_feature_name)
    df = df.sort_values("abs_impact", ascending=False)

    # SHAP only explains the machine-learning probability layer.
    # If the final decision came from the clinical safety layer, the title must
    # make that distinction clear for academic and clinical defensibility.
    shap_title = (
        "SHAP Feature Impact on Model Probability"
        if risk_basis == "clinical_override"
        else "SHAP Feature Impact on Severe Malaria Prediction"
    )

    if chart_type == "Horizontal Bar Chart":
        fig = px.bar(
            df,
            x="impact",
            y="feature_label",
            orientation="h",
            title=shap_title,
            labels={
                "impact": "SHAP Impact",
                "feature_label": "Clinical Feature"
            }
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Vertical Bar Chart":
        fig = px.bar(
            df,
            x="feature_label",
            y="abs_impact",
            title="Absolute SHAP Contribution Ranking",
            labels={
                "abs_impact": "Absolute SHAP Impact",
                "feature_label": "Clinical Feature"
            }
        )
        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Clinical Ranking Table":
        table = df[["feature_label", "impact", "direction"]].rename(
            columns={
                "feature_label": "Clinical Feature",
                "impact": "SHAP Contribution",
                "direction": "Effect on Severe Risk"
            }
        )
        st.dataframe(table, width="stretch")

    elif chart_type == "Pie Chart":
        fig = px.pie(
            df,
            names="feature_label",
            values="abs_impact",
            title="Relative SHAP Contribution Share"
        )
        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Radar Chart":
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=df["abs_impact"],
                theta=df["feature_label"],
                fill="toself",
                name="SHAP Impact"
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=False,
            title="Radar View of Feature Contributions"
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
                textposition="outside"
            )
        )
        fig.update_layout(
            title="Waterfall-style SHAP Contribution",
            xaxis_title="Clinical Feature",
            yaxis_title="Impact on Model Probability"
        )
        st.plotly_chart(fig, width="stretch")


# ------------------------------------------------
# HEADER / HERO SECTION
# ------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🧬 Malaria Severity AI Dashboard</div>
        <div class="hero-subtitle">Explainable Hybrid Clinical Decision Support System</div>
        <div class="hero-text">
            Powered by RandomForest V3, Clinical Safety Layer, and SHAP Explainability.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "⚠️ Research Prototype Disclaimer: This dashboard is for academic, research, "
    "and demonstration use only. It is not a medical device and must not be used "
    "as a final diagnosis. Always confirm with qualified clinical evaluation and "
    "laboratory testing."
)


# ------------------------------------------------
# MULTI-TAB DASHBOARD LAYOUT
# ------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🩺 Clinical Prediction",
        "📊 Explainability Analytics",
        "📚 Research Dashboard",
        "⚙️ Model/API Information"
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
            st.slider(
                "Age",
                min_value=1,
                max_value=100,
                key="age"
            )

            st.radio(
                "Sex",
                ["Female", "Male"],
                horizontal=True,
                key="sex_label"
            )

        with col_b:
            st.info(
                "Use the grouped controls below to quickly simulate patient symptom profiles. "
                "This dashboard is designed for clinical AI demonstration, research testing, "
                "and explainable decision support."
            )

    with st.container(border=True):
        st.subheader("🧾 Symptom Selection")

        control_col1, control_col2 = st.columns([1, 1])

        with control_col1:
            st.checkbox(
                "✅ Select ALL Symptoms",
                key="select_all_symptoms",
                on_change=apply_global_select_all
            )

        with control_col2:
            if st.button("🔄 Reset Patient Form", width="stretch"):
                reset_dashboard()
                st.rerun()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌡️ General Symptoms")

            st.checkbox(
                "Select All General Symptoms",
                key="select_all_general",
                on_change=apply_general_select_all
            )

            for symptom in GENERAL_SYMPTOMS:
                st.checkbox(
                    format_feature_name(symptom),
                    key=symptom
                )

        with col2:
            st.markdown("### 🚨 Severe Malaria Indicators")

            st.checkbox(
                "Select All Severe Indicators",
                key="select_all_severe",
                on_change=apply_severe_select_all
            )

            for symptom in SEVERE_INDICATORS:
                st.checkbox(
                    format_feature_name(symptom),
                    key=symptom
                )

        st.divider()

        if st.button("🔍 Analyze Patient", width="stretch"):
            sex_value = 1 if st.session_state.sex_label == "Male" else 0

            payload = {
                "age": st.session_state.age,
                "sex": sex_value,
            }

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

            probability = result.get("probability_severe", 0) * 100
            risk = result.get("severity_risk", "UNKNOWN")
            risk_icon = get_risk_icon(risk)
            decision_source = result.get("risk_basis_display") or format_decision_source(
                result.get("risk_basis", "")
            )
            model_prediction = result.get("model_label", "Unavailable")

            with c1:
                render_metric_card("Final Clinical Decision", f"🩺 {result.get('label')}")

            with c2:
                render_metric_card("Model Prediction", f"🧠 {model_prediction}")

            with c3:
                render_metric_card("Risk Level", f"{risk_icon} {risk}")

            with c4:
                render_metric_card("Probability", f"{probability:.2f}%")

            with c5:
                render_metric_card("Decision Source", decision_source)

            with c6:
                # Prediction Certainty is supplied by the backend using
                # clinician-friendly certainty banding.
                # This avoids confusing values near 50% probability.
                prediction_certainty = result.get("prediction_certainty", 0) * 100

                render_metric_card("Prediction Certainty", f"{prediction_certainty:.1f}%")

            # ---------------------------------------------------
            # HYBRID DECISION REASONING BOX
            # ---------------------------------------------------
            # This explains whether the final decision came from the
            # model probability layer, baseline safety assessment, or
            # clinical safety layer.
            # ---------------------------------------------------
            hybrid_reasoning = result.get("hybrid_reasoning", "")

            if hybrid_reasoning:
                st.info(
                    f"🧠 Hybrid Decision Reasoning: {hybrid_reasoning}"
                )

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

            chart_type = st.selectbox(
                "Select Explainability Visualization",
                [
                    "Horizontal Bar Chart",
                    "Vertical Bar Chart",
                    "Clinical Ranking Table",
                    "Pie Chart",
                    "Radar Chart",
                    "Waterfall Style"
                ]
            )

            # Backend risk basis tells the UI whether SHAP is explaining
            # the direct model prediction or only the model-probability layer
            # behind a clinical safety layer.
            risk_basis = result.get("risk_basis", "")

            render_explainability_chart(df_shap, chart_type, risk_basis)

            # ---------------------------------------------------
            # SHAP TRANSPARENCY NOTE
            # ---------------------------------------------------
            # Makes the boundary between model explainability and
            # clinical safety rules explicit.
            # ---------------------------------------------------
            st.info(
                "Note: SHAP explains machine-learning probability patterns only. "
                "Final clinical decisions may additionally include rule-based "
                "clinical safety checks for high-risk symptoms."
            )

            st.markdown("### 🧠 AI Explanation Summary")
            st.info(result.get("explanation_summary", "No explanation summary available."))

            st.markdown("### 🏥 Clinical Safety Summary")
            st.warning(result.get("clinical_summary", "No clinical summary available."))

            with st.expander("View full SHAP values"):
                df_all = pd.DataFrame(result.get("all_shap_values", []))

                if not df_all.empty:
                    # ------------------------------------------
                    # CLINICAL FEATURE FORMATTING
                    # ------------------------------------------
                    df_all["Clinical Feature"] = df_all["feature"].apply(
                        format_feature_name
                    )

                    # ------------------------------------------
                    # ADD CLINICAL STATUS COLUMN
                    # ------------------------------------------
                    # This improves explainability transparency
                    # by showing whether the symptom was:
                    # - Present
                    # - Absent
                    # - Baseline Variable
                    # ------------------------------------------
                    def determine_feature_status(feature_name):
                        """
                        Determines whether a feature was clinically present,
                        absent, or a baseline demographic variable.
                        """

                        # Demographic baseline variables
                        if feature_name in ["age", "sex"]:
                            return "Baseline Variable"

                        # Symptom selected by clinician/user
                        if st.session_state.payload:
                            if st.session_state.payload.get(feature_name, 0) == 1:
                                return "Present"

                        # Symptom not selected
                        return "Absent"


                    df_all["Clinical Status"] = df_all["feature"].apply(
                        determine_feature_status
                    )

                    # ------------------------------------------
                    # RENAME COLUMNS FOR CLINICAL DISPLAY
                    # ------------------------------------------
                    df_all = df_all.rename(
                        columns={
                            "impact": "SHAP Contribution",
                            "direction": "Effect on Model Severe-Risk Probability"
                        }
                    )

                    # ------------------------------------------
                    # SORT BY ABSOLUTE SHAP IMPACT
                    # ------------------------------------------
                    df_all["Absolute Contribution"] = (
                        df_all["SHAP Contribution"].abs()
                    )

                    df_all = df_all.sort_values(
                        "Absolute Contribution",
                        ascending=False
                    )

                    # Sort by absolute SHAP contribution while preserving
                    # the original signed contribution for interpretation.
                    df_all["Absolute Contribution"] = df_all["SHAP Contribution"].abs()

                    df_all = df_all.sort_values(
                        "Absolute Contribution",
                        ascending=False
                    )

                    st.dataframe(
                        df_all[
                            [
                                "Clinical Feature",
                                "SHAP Contribution",
                                "Effect on Model Severe-Risk Probability",
                                "Clinical Status"
                            ]
                        ],
                        width="stretch"
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
            **hybrid explainable clinical decision support system**.

            The key research contribution is not only prediction, but the design of a
            trustworthy healthcare AI architecture that combines:

            - Machine learning probability estimation
            - Clinical safety override rules
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
            "Research Note: The model has modest statistical performance, "
            "but the hybrid clinical safety layer improves safety behavior "
            "for high-risk symptom patterns."
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
            - **Model:** RandomForestClassifier V3
            - **Decision Logic:** Model probability + clinical safety layer
            - **Explainability:** SHAP TreeExplainer
            - **Backend:** FastAPI
            - **Frontend:** Streamlit
            - **Visualization:** Plotly
            """
        )

    with st.container(border=True):
        st.subheader("📦 Current API Payload")

        if st.session_state.payload:
            st.json(st.session_state.payload)
        else:
            st.info("No prediction payload yet.")


# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown(
    f"""
    <div class="app-footer">
        © {CURRENT_YEAR} ADA Global Academy Data Science Scholar | Developed by Basil Emeokoro. All rights reserved.<br>
        Research prototype for safety-aware explainable hybrid clinical decision support.<br>
        <a href="#top" class="top-link">⬆ Return to Top</a>
    </div>
    """,
    unsafe_allow_html=True
)
