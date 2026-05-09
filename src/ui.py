"""
ui.py — Malaria Severity AI Dashboard
-------------------------------------
Streamlit frontend for the Flask malaria severity prediction API.

Updated for clean standardized feature names:
age, sex, fever, cold, rigor, fatigue, headache, bitter_tongue,
vomiting, diarrhea, convulsion, anemia, jaundice, coca_cola_urine,
hypoglycemia, prostration, hyperpyrexia
"""

import streamlit as st
import requests
import pandas as pd

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Malaria AI",
    layout="wide"
)

# -------------------------------
# CUSTOM STYLING
# -------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    font-weight: 700;
    color: #111827;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #334155;
}

div[data-testid="stMetric"] label {
    color: #cbd5f5 !important;
    font-size: 14px !important;
}

div[data-testid="stMetric"] div {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

.stProgress > div > div > div > div {
    background-color: #22c55e;
}

.status-bar {
    background: linear-gradient(90deg, #1e3a8a, #0f172a);
    padding: 12px;
    border-radius: 10px;
    color: white;
    font-weight: 500;
}

.high-risk {
    background-color: #fee2e2;
    color: #b91c1c;
    padding: 12px;
    border-radius: 8px;
}

.medium-risk {
    background-color: #fef3c7;
    color: #92400e;
    padding: 12px;
    border-radius: 8px;
}

.low-risk {
    background-color: #dcfce7;
    color: #166534;
    padding: 12px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
# 🦠 Malaria Severity AI Dashboard
### Clinical Decision Support System
---
""")

st.markdown("""
<div class="status-bar">
<strong>System Status:</strong> AI Model Active | API Connected | Ready for Diagnosis
</div>
""", unsafe_allow_html=True)

st.divider()

# -------------------------------
# INPUT SECTION
# -------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🧑 Patient Info")

    age = st.slider("Age", 0, 100, 25)

    sex_label = st.radio("Sex", ["Female", "Male"])
    sex = 1 if sex_label == "Male" else 0

    st.subheader("🩺 Primary Symptoms")

    fever = st.checkbox("Fever")
    cold = st.checkbox("Cold")
    rigor = st.checkbox("Rigor")
    fatigue = st.checkbox("Fatigue")
    headache = st.checkbox("Headache")

with col2:
    st.subheader("⚠️ Advanced Symptoms")

    bitter_tongue = st.checkbox("Bitter Tongue")
    vomiting = st.checkbox("Vomiting")
    diarrhea = st.checkbox("Diarrhea")
    convulsion = st.checkbox("Convulsion")
    anemia = st.checkbox("Anemia")
    jaundice = st.checkbox("Jaundice")
    coca_cola_urine = st.checkbox("Coca-cola Urine")

st.subheader("🚨 Critical Indicators")

col3, col4, col5 = st.columns(3)

with col3:
    hypoglycemia = st.checkbox("Hypoglycemia")

with col4:
    prostration = st.checkbox("Prostration")

with col5:
    hyperpyrexia = st.checkbox("Hyperpyrexia")

st.divider()

# -------------------------------
# DATA PAYLOAD
# -------------------------------
# These field names MUST match the cleaned dataset and Flask API exactly.
data = {
    "age": int(age),
    "sex": int(sex),
    "fever": int(fever),
    "cold": int(cold),
    "rigor": int(rigor),
    "fatigue": int(fatigue),
    "headache": int(headache),
    "bitter_tongue": int(bitter_tongue),
    "vomiting": int(vomiting),
    "diarrhea": int(diarrhea),
    "convulsion": int(convulsion),
    "anemia": int(anemia),
    "jaundice": int(jaundice),
    "coca_cola_urine": int(coca_cola_urine),
    "hypoglycemia": int(hypoglycemia),
    "prostration": int(prostration),
    "hyperpyrexia": int(hyperpyrexia),
}

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("🔍 Analyze Patient"):

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=data,
            timeout=10
        )

        result = response.json()

        if response.status_code != 200:
            st.error("API returned an error.")
            st.json(result)
            st.stop()

        prob = float(result["probability_severe"])
        risk_level = result["severity_risk"]

        st.divider()

        # -------------------------------
        # DIAGNOSIS SUMMARY
        # -------------------------------
        st.subheader("🧾 Diagnosis Summary")

        colA, colB, colC = st.columns(3)

        colA.metric("Diagnosis", result["label"])
        colB.metric("Risk Level", risk_level)
        colC.metric("Probability", f"{prob:.2%}")

        # -------------------------------
        # RISK INDICATOR
        # -------------------------------
        st.markdown("### 🚦 Risk Indicator")

        if risk_level == "HIGH":
            st.markdown(
                '<div class="high-risk">🔴 HIGH RISK – Immediate attention required</div>',
                unsafe_allow_html=True
            )

        elif risk_level == "MEDIUM":
            st.markdown(
                '<div class="medium-risk">🟠 MODERATE RISK – Monitor closely</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                '<div class="low-risk">🟢 LOW RISK – Stable condition</div>',
                unsafe_allow_html=True
            )

        st.progress(prob)

        # -------------------------------
        # EXPLAINABILITY SECTION
        # -------------------------------
        st.subheader("🔍 Key Contributing Factors")

        if "top_contributors" in result and result["top_contributors"]:

            for item in result["top_contributors"]:
                st.write(
                    f"• {item['feature']} "
                    f"(impact: {item['importance']:.3f})"
                )

            st.subheader("📊 Feature Contribution Chart")

            df_chart = pd.DataFrame(result["top_contributors"])

            df_chart = df_chart.sort_values(
                by="importance",
                ascending=True
            )

            st.bar_chart(
                df_chart.set_index("feature")
            )

        else:
            st.info("No explainability data returned by API.")

        # -------------------------------
        # CLINICAL RECOMMENDATION
        # -------------------------------
        st.markdown("""
---
💡 **Clinical Recommendation:**  

- High Risk → Immediate intervention  
- Moderate Risk → Observation & tests  
- Low Risk → Routine monitoring  
""")

        # -------------------------------
        # DEBUG PAYLOAD VIEW
        # -------------------------------
        with st.expander("🔎 Debug: View API Payload"):
            st.json(data)

    except Exception as e:
        st.error(
            "🚫 API not reachable. Ensure Flask server is running."
        )
        st.exception(e)