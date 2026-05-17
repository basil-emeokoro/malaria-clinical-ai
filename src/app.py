"""
app.py — Malaria Severity Prediction API (V5)
------------------------------------------------
Hybrid Explainable Clinical ML API

Architecture:
✅ V3 RandomForest model
✅ Clinical override engine
✅ SHAP TreeExplainer
✅ FastAPI backend
✅ Explainability-ready JSON response
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
    version="5.0.0",
    description="Hybrid explainable clinical decision support API for malaria severity prediction."
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
        "architecture": "V3 RandomForest + Clinical Override + SHAP",
        "features": FEATURES,
        "target": "severe_malaria"
    }


# ------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------
@app.post("/predict")
def predict(data: PatientData):

    # Convert Pydantic object into dictionary
    body = data.model_dump()

    # Create dataframe in exact training feature order
    X = pd.DataFrame(
        [[float(body[f]) for f in FEATURES]],
        columns=FEATURES
    )

    # -------------------------------
    # ML PREDICTION
    # -------------------------------
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    label = (
        "Severe Malaria"
        if prediction == 1
        else "Not Severe Malaria"
    )

    # -------------------------------
    # CLINICAL OVERRIDE ENGINE
    # -------------------------------
    critical_symptoms = [
        "convulsion",
        "hypoglycemia",
        "prostration",
        "hyperpyrexia",
        "jaundice",
        "coca_cola_urine"
    ]

    active_critical_symptoms = [
        symptom for symptom in critical_symptoms
        if int(body.get(symptom, 0)) == 1
    ]

    critical_flags = len(active_critical_symptoms)

    if critical_flags >= 3:
        severity_risk = "HIGH"
        risk_basis = "clinical_override"

    elif probability >= 0.55:
        severity_risk = "HIGH"
        risk_basis = "model_probability"

    elif probability >= 0.35:
        severity_risk = "MEDIUM"
        risk_basis = "model_probability"

    else:
        severity_risk = "LOW"
        risk_basis = "model_probability"

    # If rule engine escalates to HIGH, final display label should reflect severe concern
    final_label = (
        "Severe Malaria"
        if severity_risk == "HIGH" or prediction == 1
        else "Not Severe Malaria"
    )

    # -------------------------------
    # SHAP EXPLAINABILITY
    # -------------------------------
    shap_values = explainer.shap_values(X)

    # For binary classification, class index 1 = severe malaria
    severe_shap_values = shap_values[1][0]

    shap_contributors = []

    for feature, impact in zip(FEATURES, severe_shap_values):
        shap_contributors.append({
            "feature": feature,
            "impact": round(float(impact), 5),
            "direction": (
                "increases severe risk"
                if impact > 0
                else "reduces severe risk"
                if impact < 0
                else "neutral"
            )
        })

    # Sort by absolute SHAP impact
    shap_contributors = sorted(
        shap_contributors,
        key=lambda item: abs(item["impact"]),
        reverse=True
    )

    top_contributors = shap_contributors[:7]

    # -------------------------------
    # HUMAN-READABLE EXPLANATION
    # -------------------------------
    positive_features = [
        item["feature"]
        for item in top_contributors
        if item["impact"] > 0
    ]

    negative_features = [
        item["feature"]
        for item in top_contributors
        if item["impact"] < 0
    ]

    if positive_features:
        explanation_summary = (
            "The prediction was mainly influenced by "
            + ", ".join(positive_features[:3])
            + ", which pushed the model toward severe malaria risk."
        )
    else:
        explanation_summary = (
            "No strong positive SHAP contributor was detected for severe malaria risk."
        )

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

    # -------------------------------
    # RESPONSE
    # -------------------------------
    return {
        "prediction": prediction,
        "label": final_label,
        "model_label": label,
        "probability_severe": round(probability, 4),
        "severity_risk": severity_risk,
        "risk_basis": risk_basis,
        "critical_flags": critical_flags,
        "active_critical_symptoms": active_critical_symptoms,
        "top_contributors": top_contributors,
        "all_shap_values": shap_contributors,
        "explanation_summary": explanation_summary,
        "clinical_summary": clinical_summary
    }