import streamlit as st
import requests
import pandas as pd

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
# This sets the browser tab name and layout
st.set_page_config(page_title="Malaria AI", layout="wide")

# -------------------------------
# CUSTOM STYLING (CSS)
# -------------------------------
# This improves visual appearance using custom CSS
st.markdown("""
<style>

/* Page spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1, h2, h3 {
    font-weight: 700;
    color: #111827;
}

/* Metric cards (Diagnosis / Risk / Probability) */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #334155;
}

/* Metric labels */
div[data-testid="stMetric"] label {
    color: #cbd5f5 !important;
    font-size: 14px !important;
}

/* Metric values (force bright text) */
div[data-testid="stMetric"] div {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* Progress bar color */
.stProgress > div > div > div > div {
    background-color: #22c55e;
}

/* Status bar styling */
.status-bar {
    background: linear-gradient(90deg, #1e3a8a, #0f172a);
    padding: 12px;
    border-radius: 10px;
    color: white;
    font-weight: 500;
}

/* Risk alert boxes */
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
# HEADER SECTION
# -------------------------------
st.markdown("""
# 🦠 Malaria Severity AI Dashboard
### Clinical Decision Support System
---
""")

# System status indicator
st.markdown("""
<div class="status-bar">
<strong>System Status:</strong> AI Model Active | API Connected | Ready for Diagnosis
</div>
""", unsafe_allow_html=True)

st.divider()

# -------------------------------
# INPUT SECTION (USER FORM)
# -------------------------------
col1, col2 = st.columns([1, 1])

# LEFT COLUMN (Basic Info)
with col1:
    st.subheader("🧑 Patient Info")

    # Age slider
    age = st.slider("Age", 0, 100, 25)

    # Sex selection
    sex = st.radio("Sex", ["Female", "Male"])

    # Primary symptoms
    st.subheader("🩺 Primary Symptoms")
    fever = st.checkbox("Fever")
    cold = st.checkbox("Cold")
    rigor = st.checkbox("Rigor")
    fatigue = st.checkbox("Fatigue")
    headache = st.checkbox("Headache")

# RIGHT COLUMN (Advanced Symptoms)
with col2:
    st.subheader("⚠️ Advanced Symptoms")
    bitter_tongue = st.checkbox("Bitter Tongue")
    vomiting = st.checkbox("Vomiting")
    diarrhea = st.checkbox("Diarrhea")
    convulsion = st.checkbox("Convulsion")
    anemia = st.checkbox("Anemia")
    jaundice = st.checkbox("Jaundice")
    cocacola_urine = st.checkbox("Coca-cola Urine")

# Critical indicators (high-risk features)
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
# DATA PREPARATION
# -------------------------------
# Convert user input into API-compatible JSON format
data = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "fever": int(fever),
    "cold": int(cold),
    "rigor": int(rigor),
    "fatigue": int(fatigue),
    "headace": int(headache),
    "bitter_tongue": int(bitter_tongue),
    "vomitting": int(vomiting),
    "diarrhea": int(diarrhea),
    "Convulsion": int(convulsion),
    "Anemia": int(anemia),
    "jundice": int(jaundice),
    "cocacola_urine": int(cocacola_urine),
    "hypoglycemia": int(hypoglycemia),
    "prostraction": int(prostration),
    "hyperpyrexia": int(hyperpyrexia),
}

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("🔍 Analyze Patient"):

    try:

        # -----------------------------------
        # Send request to Flask API backend
        # -----------------------------------
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=data
        )

        # Convert API response into JSON
        result = response.json()

        # Extract probability score
        prob = result["probability_severe"]

        st.divider()

        # -----------------------------------
        # DIAGNOSIS SUMMARY
        # -----------------------------------
        st.subheader("🧾 Diagnosis Summary")

        # Create 3 metric cards
        colA, colB, colC = st.columns(3)

        # Diagnosis card
        colA.metric(
            "Diagnosis",
            result["label"]
        )

        # Risk level card
        colB.metric(
            "Risk Level",
            result["severity_risk"]
        )

        # Probability card
        colC.metric(
            "Probability",
            f"{prob:.2%}"
        )

        # -----------------------------------
        # 🚦 RISK INDICATOR SECTION
        # -----------------------------------
        st.markdown("### 🚦 Risk Indicator")

        # Dynamic risk alerts
        if prob > 0.7:

            st.markdown(
                '''
                <div class="high-risk">
                🔴 HIGH RISK – Immediate attention required
                </div>
                ''',
                unsafe_allow_html=True
            )

        elif prob > 0.4:

            st.markdown(
                '''
                <div class="medium-risk">
                🟠 MODERATE RISK – Monitor closely
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '''
                <div class="low-risk">
                🟢 LOW RISK – Stable condition
                </div>
                ''',
                unsafe_allow_html=True
            )

        # Probability progress bar
        st.progress(prob)

        # -----------------------------------
        # 🔍 EXPLAINABILITY SECTION
        # -----------------------------------
        # This section displays:
        #
        # ✅ top contributing features
        # ✅ feature importance scores
        # ✅ visual contribution chart
        #
        # Data comes LIVE from Flask API
        # -----------------------------------

        st.subheader("🔍 Key Contributing Factors")

        # Ensure explainability exists
        if (
            "top_contributors" in result
            and result["top_contributors"]
        ):

            # Display top features
            for item in result["top_contributors"]:

                st.write(
                    f"• {item['feature']} "
                    f"(impact: {item['importance']:.3f})"
                )

            # -----------------------------------
            # 📊 FEATURE CONTRIBUTION CHART
            # -----------------------------------
            st.subheader(
                "📊 Feature Contribution Chart"
            )

            # Convert JSON -> dataframe
            df_chart = pd.DataFrame(
                result["top_contributors"]
            )

            # Sort ascending for better visuals
            df_chart = df_chart.sort_values(
                by="importance",
                ascending=True
            )

            # Render chart
            st.bar_chart(
                df_chart.set_index("feature")
            )

        else:

            st.info(
                "No explainability data returned by API."
            )

        # -----------------------------------
        # 💡 CLINICAL RECOMMENDATION
        # -----------------------------------
        st.markdown("""
---
💡 **Clinical Recommendation:**  

- High Risk → Immediate intervention  
- Moderate Risk → Observation & tests  
- Low Risk → Routine monitoring  
""")

    # -----------------------------------
    # ERROR HANDLING
    # -----------------------------------
    except Exception as e:

        st.error(
            "🚫 API not reachable. "
            "Ensure Flask server is running."
        )

        # Optional debugging output
        st.exception(e)