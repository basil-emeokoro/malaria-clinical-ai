"""
app.py — Malaria Severity Prediction API (V5.1)
------------------------------------------------
Hybrid Explainable Clinical ML API

Architecture:
- V3 RandomForest model
- Clinical safety override engine
- SHAP TreeExplainer
- FastAPI backend
- Prediction certainty banding
- Override-aware explanation logic
"""

import joblib
import shap
import pandas as pd

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
        "architecture": "V3 RandomForest + Clinical Override + SHAP + Prediction Certainty",
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

    # ------------------------------------------------
    # HEALTHCARE-FRIENDLY PREDICTION CERTAINTY
    # ------------------------------------------------
    if probability_severe < 0.30:
        prediction_certainty = 0.90

    elif probability_severe <= 0.70:
        prediction_certainty = 0.65

    else:
        prediction_certainty = 0.90

    # ------------------------------------------------
    # SHAP EXPLAINABILITY
    # ------------------------------------------------
    shap_values = explainer.shap_values(X)

    # Class index 1 explains severe malaria probability
    severe_shap_values = shap_values[1][0]

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
    if risk_basis == "clinical_override":
        explanation_summary = (
            "The final prediction was upgraded to Severe Malaria due to "
            "critical clinical indicators including "
            f"{', '.join(active_critical_symptoms)}. "
            "The override mechanism prioritizes patient safety over model probability."
        )

    elif active_symptom_count == 0:
        explanation_summary = (
            "No clinically observed malaria symptoms were active. "
            "The prediction was based primarily on baseline demographic factors "
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
                "The prediction was mainly influenced by "
                + ", ".join(active_positive_features[:3])
                + ", which increased severe malaria risk."
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
            "Clinical override considered the following critical indicators: "
            + ", ".join(active_critical_symptoms)
            + "."
        )
    else:
        clinical_summary = (
            "No critical clinical override indicators were active."
        )

    # ------------------------------------------------
    # HYBRID DECISION REASONING
    # ------------------------------------------------
    if risk_basis == "clinical_override":
        hybrid_reasoning = (
            f"The statistical model estimated a {probability_severe:.2%} severe malaria probability. "
            f"However, the final diagnosis was escalated to Severe Malaria because "
            f"{critical_flags} clinically critical indicator(s) were detected: "
            f"{', '.join(active_critical_symptoms)}. "
            "The clinical safety layer prioritizes patient protection over statistical uncertainty."
        )
    else:
        hybrid_reasoning = (
            f"The statistical model estimated a {probability_severe:.2%} severe malaria probability. "
            f"The final decision was based on {risk_basis.replace('_', ' ')}."
        )

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
        "severity_risk": severity_risk,
        "risk_basis": risk_basis,
        "critical_flags": critical_flags,
        "active_symptom_count": active_symptom_count,
        "active_symptoms": active_symptoms,
        "active_critical_symptoms": active_critical_symptoms,
        "top_contributors": top_contributors,
        "all_shap_values": shap_contributors,
        "explanation_summary": explanation_summary,
        "clinical_summary": clinical_summary,
        "hybrid_reasoning": hybrid_reasoning
    }
