"""
app.py — Malaria Severity Prediction API (V5.1)
------------------------------------------------
Hybrid Explainable Clinical ML API

Architecture:
- V3 RandomForest model
- Clinical safety assessment layer
- SHAP TreeExplainer
- FastAPI backend
- Prediction certainty banding
- Safety-aware explanation logic
"""

import joblib
import shap
import pandas as pd
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel


# ------------------------------------------------
# LOAD MODEL + FEATURE SCHEMA
# ------------------------------------------------
model = joblib.load("model/model_v3.joblib")
FEATURES = joblib.load("model/features_v3.joblib")

# SHAP explainer for RandomForest model
explainer = shap.TreeExplainer(model)


# ------------------------------------------------
# FASTAPI APP
# ------------------------------------------------
app = FastAPI(
    title="Malaria Severity Prediction API",
    version="5.1.0",
    description="Safety-aware explainable hybrid clinical decision support API for malaria severity prediction."
)


# ------------------------------------------------
# INPUT SCHEMA
# ------------------------------------------------
class PatientData(BaseModel):
    age: int
    sex: int
    fever: int
    cold: int
    rigor: int
    fatigue: int
    headache: int
    bitter_tongue: int
    vomiting: int
    diarrhea: int
    convulsion: int
    anemia: int
    jaundice: int
    coca_cola_urine: int
    hypoglycemia: int
    prostration: int
    hyperpyrexia: int


# ------------------------------------------------
# ROOT ENDPOINT
# ------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Malaria Severity Prediction API is active."
    }


# ------------------------------------------------
# HEALTH ENDPOINT
# ------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": "v3",
        "api_version": "5.1.0",
        "shap_enabled": True
    }


# ------------------------------------------------
# INFO ENDPOINT
# ------------------------------------------------
@app.get("/info")
def info():
    return {
        "model": "RandomForestClassifier",
        "model_version": "v3",
        "architecture": "V3 RandomForest + Clinical Safety Layer + SHAP + Prediction Certainty",
        "features": FEATURES,
        "target": "severe_malaria"
    }


# ------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------
@app.post("/predict")
def predict(data: PatientData):
    """
    Receives patient symptoms, performs ML prediction,
    applies clinical safety override rules, calculates SHAP values,
    and returns explainable clinical decision support output.
    """

    # ------------------------------------------------
    # CONVERT REQUEST BODY TO MODEL INPUT
    # ------------------------------------------------
    body = data.model_dump()

    X = pd.DataFrame(
        [[float(body[f]) for f in FEATURES]],
        columns=FEATURES
    )

    # ------------------------------------------------
    # MACHINE LEARNING PREDICTION
    # ------------------------------------------------
    model_prediction = int(model.predict(X)[0])
    probability_severe = float(model.predict_proba(X)[0][1])

    model_label = (
        "Severe Malaria"
        if model_prediction == 1
        else "Not Severe Malaria"
    )

    # ------------------------------------------------
    # CLINICAL FEATURE GROUPS
    # ------------------------------------------------
    critical_symptoms = [
        "convulsion",
        "hypoglycemia",
        "prostration",
        "hyperpyrexia",
        "jaundice",
        "coca_cola_urine"
    ]

    symptom_features = [
        "fever",
        "cold",
        "rigor",
        "fatigue",
        "headache",
        "bitter_tongue",
        "vomiting",
        "diarrhea",
        "convulsion",
        "anemia",
        "jaundice",
        "coca_cola_urine",
        "hypoglycemia",
        "prostration",
        "hyperpyrexia"
    ]

    active_symptoms = [
        symptom for symptom in symptom_features
        if int(body.get(symptom, 0)) == 1
    ]

    active_critical_symptoms = [
        symptom for symptom in critical_symptoms
        if int(body.get(symptom, 0)) == 1
    ]

    active_symptom_count = len(active_symptoms)
    critical_flags = len(active_critical_symptoms)

    # ------------------------------------------------
    # HYBRID CLINICAL RISK DECISION LOGIC
    # ------------------------------------------------
    if active_symptom_count == 0:
        severity_risk = "LOW"
        risk_basis = "clinical_baseline_guardrail"

    elif critical_flags >= 1:
        severity_risk = "HIGH"
        risk_basis = "clinical_override"

    elif probability_severe >= 0.75:
        severity_risk = "HIGH"
        risk_basis = "model_probability"

    elif probability_severe >= 0.50:
        severity_risk = "MODERATE"
        risk_basis = "model_probability"

    else:
        severity_risk = "LOW"
        risk_basis = "model_probability"

    # ------------------------------------------------
    # FINAL CLINICAL DECISION SYNCHRONIZATION
    # ------------------------------------------------
    if risk_basis == "clinical_override":
        final_prediction = 1

    elif severity_risk == "HIGH":
        final_prediction = 1

    else:
        final_prediction = model_prediction

    final_label = (
        "Severe Malaria"
        if final_prediction == 1
        else "Not Severe Malaria"
    )

    clinical_override_triggered = risk_basis == "clinical_override"
    safety_escalated = (
        clinical_override_triggered
        and model_prediction != final_prediction
    )
    safety_confirmed = (
        clinical_override_triggered
        and model_prediction == final_prediction
        and critical_flags > 0
    )

    if safety_escalated:
        risk_basis_display = "Safety Escalation Active"
        clinical_safety_status = "escalated"

    elif safety_confirmed:
        risk_basis_display = "Clinical Safety Confirmed"
        clinical_safety_status = "confirmed"

    elif risk_basis == "clinical_baseline_guardrail":
        risk_basis_display = "Baseline Clinical Safety Assessment"
        clinical_safety_status = "baseline_assessment"

    elif risk_basis == "model_probability":
        risk_basis_display = "Model Probability"
        clinical_safety_status = "model_probability"

    else:
        risk_basis_display = risk_basis.replace("_", " ").title()
        clinical_safety_status = "not_applicable"

    # ------------------------------------------------
    # HEALTHCARE-FRIENDLY PREDICTION CERTAINTY
    # ------------------------------------------------
    if critical_flags >= 4:
        prediction_certainty = 0.95
        prediction_certainty_basis = (
            "High certainty because multiple critical clinical indicators were present."
        )

    elif critical_flags >= 2:
        prediction_certainty = 0.88
        prediction_certainty_basis = (
            "Elevated certainty because more than one critical clinical indicator was present."
        )

    elif probability_severe > 0.75:
        prediction_certainty = 0.85
        prediction_certainty_basis = (
            "Elevated certainty because the model estimated high severe-malaria probability."
        )

    elif probability_severe > 0.50:
        prediction_certainty = 0.75
        prediction_certainty_basis = (
            "Moderate certainty because the model probability was above the severe-risk threshold."
        )

    else:
        prediction_certainty = 0.65
        prediction_certainty_basis = (
            "Baseline certainty because severe-risk evidence was limited or mixed."
        )

    # ------------------------------------------------
    # SHAP EXPLAINABILITY
    # ------------------------------------------------
    shap_values = explainer.shap_values(X)

    # Class index 1 explains severe malaria probability.
    # Supports both older SHAP list output and newer 3D ndarray output.
    if isinstance(shap_values, list):
        severe_shap_values = shap_values[1][0]
    else:
        severe_shap_values = shap_values[0, :, 1]

    shap_contributors = []

    for feature, impact in zip(FEATURES, severe_shap_values):
        impact_value = float(impact)

        shap_contributors.append({
            "feature": feature,
            "impact": round(impact_value, 5),
            "direction": (
                "increases severe risk"
                if impact_value > 0
                else "reduces severe risk"
                if impact_value < 0
                else "neutral"
            )
        })

    shap_contributors = sorted(
        shap_contributors,
        key=lambda item: abs(item["impact"]),
        reverse=True
    )

    top_contributors = shap_contributors[:7]

    # ------------------------------------------------
    # OVERRIDE-AWARE EXPLANATION SUMMARY
    # ------------------------------------------------
    if safety_escalated:
        explanation_summary = (
            "The final clinical decision was escalated to Severe Malaria because "
            "critical clinical indicators were present, including "
            f"{', '.join(active_critical_symptoms)}. "
            "This safety layer prioritizes clinically significant warning signs when they conflict with model output."
        )

    elif safety_confirmed:
        explanation_summary = (
            "The model prediction and clinical safety assessment both supported Severe Malaria. "
            "Critical indicators including "
            f"{', '.join(active_critical_symptoms)} "
            "confirmed the high-risk interpretation."
        )

    elif active_symptom_count == 0:
        explanation_summary = (
            "No clinically observed malaria symptoms were active. "
            "The result reflects a Baseline Clinical Safety Assessment using demographic inputs "
            "and learned model patterns."
        )

    else:
        active_positive_features = [
            item["feature"].replace("_", " ")
            for item in top_contributors
            if item["impact"] > 0
            and (
                item["feature"] in ["age", "sex"]
                or int(body.get(item["feature"], 0)) == 1
            )
        ]

        if active_positive_features:
            explanation_summary = (
                "The prediction was mainly influenced by observed clinical features including "
                + ", ".join(active_positive_features[:3])
                + ", which collectively increased estimated severe malaria risk."
            )
        else:
            explanation_summary = (
                "Observed symptom profile showed relatively low severe-risk "
                "indicators according to the model."
            )

    # ------------------------------------------------
    # CLINICAL SUMMARY
    # ------------------------------------------------
    if active_critical_symptoms:
        clinical_summary = (
            "Clinical safety assessment identified the following critical indicators: "
            + ", ".join(active_critical_symptoms)
            + "."
        )
    else:
        clinical_summary = (
            "No critical clinical safety indicators were active."
        )

    # ------------------------------------------------
    # HYBRID DECISION REASONING
    # ------------------------------------------------
    if safety_escalated:
        hybrid_reasoning = (
            f"The statistical model estimated a {probability_severe:.2%} severe malaria probability. "
            f"The final diagnosis was escalated to Severe Malaria because "
            f"{critical_flags} clinically critical indicator(s) were detected: "
            f"{', '.join(active_critical_symptoms)}. "
            f"{prediction_certainty_basis}"
        )

    elif safety_confirmed:
        hybrid_reasoning = (
            f"The statistical model estimated a {probability_severe:.2%} severe malaria probability "
            "and already predicted Severe Malaria. "
            f"The clinical safety assessment confirmed this interpretation because "
            f"{critical_flags} clinically critical indicator(s) were detected: "
            f"{', '.join(active_critical_symptoms)}. "
            f"{prediction_certainty_basis}"
        )

    elif risk_basis == "clinical_baseline_guardrail":
        hybrid_reasoning = (
            f"The statistical model estimated a {probability_severe:.2%} severe malaria probability. "
            "The final decision used the Baseline Clinical Safety Assessment because no active malaria symptoms were recorded. "
            f"{prediction_certainty_basis}"
        )

    else:
        hybrid_reasoning = (
            f"The statistical model estimated a {probability_severe:.2%} severe malaria probability. "
            f"The final decision was based on {risk_basis_display}. "
            f"{prediction_certainty_basis}"
        )

    prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------
    # RESPONSE
    # ------------------------------------------------
    return {
        "prediction": final_prediction,
        "label": final_label,
        "model_prediction": model_prediction,
        "model_label": model_label,
        "probability_severe": round(probability_severe, 4),
        "prediction_certainty": prediction_certainty,
        "prediction_certainty_basis": prediction_certainty_basis,
        "severity_risk": severity_risk,
        "risk_basis": risk_basis,
        "risk_basis_display": risk_basis_display,
        "clinical_safety_status": clinical_safety_status,
        "critical_flags": critical_flags,
        "active_symptom_count": active_symptom_count,
        "active_symptoms": active_symptoms,
        "active_critical_symptoms": active_critical_symptoms,
        "top_contributors": top_contributors,
        "all_shap_values": shap_contributors,
        "explanation_summary": explanation_summary,
        "clinical_summary": clinical_summary,
        "hybrid_reasoning": hybrid_reasoning,
        "prediction_timestamp": prediction_timestamp
    }
