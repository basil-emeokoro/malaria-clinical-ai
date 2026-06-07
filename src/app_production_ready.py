"""
app.py — Malaria Severity Prediction API
------------------------------------------------
Production-ready hybrid explainable clinical decision support API.

Purpose:
This FastAPI backend receives malaria symptom data, runs the trained
RandomForest model, applies a clinical safety override layer, generates
SHAP explainability values, and returns an audit-friendly response.

Important clinical note:
This application is a research and demonstration system. It does not replace
professional medical diagnosis, clinical examination, or laboratory testing.
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
import shap

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ------------------------------------------------
# LOAD MODEL + FEATURE SCHEMA
# ------------------------------------------------
model = joblib.load("model/model_v3.joblib")
FEATURES = joblib.load("model/features_v3.joblib")
explainer = shap.TreeExplainer(model)


# ------------------------------------------------
# FASTAPI APP
# ------------------------------------------------
app = FastAPI(
    title="Malaria Severity Prediction API",
    version="6.0.0",
    description=(
        "Safety-aware hybrid explainable clinical decision support API "
        "for malaria severity prediction."
    ),
)


# ------------------------------------------------
# INPUT SCHEMA
# ------------------------------------------------
class PatientData(BaseModel):
    """Input schema expected by the prediction endpoint."""

    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    fever: int = Field(..., ge=0, le=1)
    cold: int = Field(..., ge=0, le=1)
    rigor: int = Field(..., ge=0, le=1)
    fatigue: int = Field(..., ge=0, le=1)
    headache: int = Field(..., ge=0, le=1)
    bitter_tongue: int = Field(..., ge=0, le=1)
    vomiting: int = Field(..., ge=0, le=1)
    diarrhea: int = Field(..., ge=0, le=1)
    convulsion: int = Field(..., ge=0, le=1)
    anemia: int = Field(..., ge=0, le=1)
    jaundice: int = Field(..., ge=0, le=1)
    coca_cola_urine: int = Field(..., ge=0, le=1)
    hypoglycemia: int = Field(..., ge=0, le=1)
    prostration: int = Field(..., ge=0, le=1)
    hyperpyrexia: int = Field(..., ge=0, le=1)


# ------------------------------------------------
# CLINICAL FEATURE GROUPS
# ------------------------------------------------
GENERAL_SYMPTOMS = [
    "fever", "cold", "rigor", "fatigue", "headache",
    "bitter_tongue", "vomiting", "diarrhea",
]

CRITICAL_SYMPTOMS = [
    "convulsion", "hypoglycemia", "prostration",
    "hyperpyrexia", "jaundice", "coca_cola_urine",
]

SEVERE_INDICATORS = [
    "convulsion", "anemia", "jaundice", "coca_cola_urine",
    "hypoglycemia", "prostration", "hyperpyrexia",
]

SYMPTOM_FEATURES = GENERAL_SYMPTOMS + SEVERE_INDICATORS


# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def pretty_feature_name(feature: str) -> str:
    """Converts snake_case feature names into readable clinical labels."""
    return feature.replace("_", " ").title()


def get_active_features(body: dict[str, Any], features: list[str]) -> list[str]:
    """Returns features whose binary value is active in the submitted payload."""
    return [feature for feature in features if int(body.get(feature, 0)) == 1]


def extract_severe_class_shap_values(shap_values: Any) -> list[float]:
    """
    Extracts SHAP values for the severe-malaria class.

    Supports common SHAP output shapes:
    - list[class][row][feature]
    - ndarray[row][feature][class]
    - ndarray[row][feature]
    """
    if isinstance(shap_values, list):
        return list(shap_values[1][0])

    if getattr(shap_values, "ndim", None) == 3:
        return list(shap_values[0, :, 1])

    return list(shap_values[0])


def build_shap_contributors(X: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Builds sorted SHAP contributor records for the severe-malaria probability.

    Direction describes model probability behaviour only, not clinical causation.
    """
    raw_shap_values = explainer.shap_values(X)
    severe_shap_values = extract_severe_class_shap_values(raw_shap_values)

    contributors = []

    for feature, impact in zip(FEATURES, severe_shap_values):
        impact_value = float(impact)
        contributors.append(
            {
                "feature": feature,
                "impact": round(impact_value, 5),
                "direction": (
                    "increases severe risk"
                    if impact_value > 0
                    else "reduces severe risk"
                    if impact_value < 0
                    else "neutral"
                ),
            }
        )

    return sorted(contributors, key=lambda item: abs(item["impact"]), reverse=True)


def calculate_severity_risk(
    probability_severe: float,
    active_symptom_count: int,
    critical_flags: int,
) -> tuple[str, str]:
    """
    Determines final risk level and decision source.

    Rules:
    - No symptoms: baseline clinical guardrail, LOW risk.
    - Any critical symptom: safety override, HIGH risk.
    - High model probability: HIGH risk.
    - Moderate probability or high symptom burden: MODERATE risk.
    - Otherwise: LOW risk.
    """
    if active_symptom_count == 0:
        return "LOW", "clinical_baseline_guardrail"

    if critical_flags > 0:
        return "HIGH", "clinical_override"

    if probability_severe >= 0.70:
        return "HIGH", "model_probability"

    if probability_severe >= 0.40 or active_symptom_count >= 4:
        return "MODERATE", "symptom_burden_guardrail"

    return "LOW", "model_probability"


def calculate_prediction_certainty(
    probability_severe: float,
    critical_flags: int,
    risk_basis: str,
) -> float:
    """
    Produces clinician-friendly prediction certainty.

    This is machine/hybrid-system certainty, not medical certainty.
    """
    if risk_basis == "clinical_override":
        return round(min(0.80 + (critical_flags * 0.05), 0.99), 2)

    if probability_severe < 0.30 or probability_severe > 0.70:
        return 0.90

    return 0.65


def build_final_decision(
    model_prediction: int,
    severity_risk: str,
    risk_basis: str,
) -> int:
    """Synchronizes the numeric prediction with the final hybrid decision."""
    if risk_basis == "clinical_override":
        return 1

    if model_prediction == 1:
        return 1

    if severity_risk == "HIGH":
        return 1

    return 0


def build_explanation_summary(
    risk_basis: str,
    active_symptoms: list[str],
    active_critical_symptoms: list[str],
    top_contributors: list[dict[str, Any]],
    body: dict[str, Any],
) -> str:
    """Builds a human-readable explanation summary."""
    if risk_basis == "clinical_override":
        return (
            "The final prediction was upgraded to Severe Malaria due to "
            "critical clinical indicators including "
            f"{', '.join(active_critical_symptoms)}. "
            "The override mechanism prioritizes patient safety over model probability."
        )

    if len(active_symptoms) == 0:
        return (
            "No clinically observed malaria symptoms were active. "
            "The prediction relied primarily on demographic baseline factors "
            "and statistical model patterns."
        )

    active_positive_features = [
        item["feature"]
        for item in top_contributors
        if item["impact"] > 0
        and (
            item["feature"] in ["age", "sex"]
            or int(body.get(item["feature"], 0)) == 1
        )
    ]

    if len(active_symptoms) >= 4:
        symptom_text = ", ".join(pretty_feature_name(symptom) for symptom in active_symptoms[:8])

        if active_positive_features:
            contributor_text = ", ".join(
                pretty_feature_name(feature) for feature in active_positive_features[:3]
            )
            return (
                f"Multiple malaria-consistent symptoms were detected, including {symptom_text}. "
                f"The model probability was mainly increased by {contributor_text}. "
                "Close clinical observation and confirmatory testing are recommended."
            )

        return (
            f"Multiple malaria-consistent symptoms were detected, including {symptom_text}. "
            "Although the model did not escalate the case to severe malaria, "
            "the symptom burden supports close clinical observation."
        )

    if active_positive_features:
        contributor_text = ", ".join(
            pretty_feature_name(feature) for feature in active_positive_features[:3]
        )
        return (
            f"The prediction was mainly influenced by {contributor_text}, "
            "which increased severe malaria risk."
        )

    return (
        "Observed symptoms were present, but the model did not identify a strong "
        "positive severe-malaria probability pattern. Clinical confirmation remains advised."
    )


def build_clinical_summary(active_critical_symptoms: list[str]) -> str:
    """Builds a short clinical safety summary."""
    if active_critical_symptoms:
        return (
            "Clinical override considered the following critical indicators: "
            + ", ".join(active_critical_symptoms)
            + "."
        )
    return "No critical clinical override indicators were active."


def build_hybrid_reasoning(
    probability_severe: float,
    final_label: str,
    risk_basis: str,
    critical_flags: int,
    active_critical_symptoms: list[str],
    active_symptom_count: int,
    severity_risk: str,
) -> str:
    """Creates a clear audit-friendly hybrid decision explanation."""
    probability_text = f"{probability_severe:.2%}"

    if risk_basis == "clinical_override":
        return (
            f"The statistical model estimated a {probability_text} severe malaria probability. "
            f"However, the final diagnosis was escalated to {final_label} because "
            f"{critical_flags} clinically critical indicator(s) were detected: "
            f"{', '.join(active_critical_symptoms)}. "
            "The clinical safety layer prioritizes patient protection over statistical uncertainty."
        )

    if risk_basis == "clinical_baseline_guardrail":
        return (
            f"The statistical model estimated a {probability_text} severe malaria probability. "
            "No clinically observed malaria symptoms were selected, so the baseline clinical "
            "guardrail treated the case as low risk while preserving the model probability "
            "for audit review."
        )

    if risk_basis == "symptom_burden_guardrail":
        return (
            f"The statistical model estimated a {probability_text} severe malaria probability. "
            f"{active_symptom_count} malaria-consistent symptom(s) were detected, so the "
            f"system classified the case as {severity_risk} risk for closer monitoring."
        )

    return (
        f"The statistical model estimated a {probability_text} severe malaria probability. "
        "The final decision was based on model probability."
    )


def build_model_warning(top_contributors: list[dict[str, Any]]) -> str:
    """Adds transparency note about clinically suspicious SHAP directions."""
    critical_features = set(CRITICAL_SYMPTOMS)
    suspicious = [
        item["feature"]
        for item in top_contributors
        if item["feature"] in critical_features and item["impact"] < 0
    ]

    if not suspicious:
        return (
            "SHAP values explain machine-learning probability patterns only. "
            "They should not be interpreted as clinical causation."
        )

    suspicious_text = ", ".join(pretty_feature_name(feature) for feature in suspicious)

    return (
        "Model validation note: SHAP assigned risk-reducing contributions to clinically "
        f"critical feature(s): {suspicious_text}. This supports the need for the clinical "
        "safety override layer and further model retraining before real-world deployment."
    )


# ------------------------------------------------
# ROOT ENDPOINT
# ------------------------------------------------
@app.get("/")
def root() -> dict[str, Any]:
    """Simple API status check."""
    return {
        "status": "running",
        "message": "Malaria Severity Prediction API is active.",
        "version": "6.0.0",
    }


# ------------------------------------------------
# HEALTH ENDPOINT
# ------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    """Health endpoint for local and cloud deployment checks."""
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": "v3",
        "shap_enabled": True,
    }


# ------------------------------------------------
# INFO ENDPOINT
# ------------------------------------------------
@app.get("/info")
def info() -> dict[str, Any]:
    """Returns model metadata and feature schema."""
    return {
        "model": "RandomForestClassifier",
        "model_version": "v3",
        "architecture": "RandomForest + Clinical Safety Override + SHAP",
        "features": FEATURES,
        "target": "severe_malaria",
    }


# ------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------
@app.post("/predict")
def predict(data: PatientData) -> dict[str, Any]:
    """
    Receives patient symptoms, performs ML prediction, applies clinical
    guardrails, calculates SHAP values, and returns explainable output.
    """
    body = data.model_dump()

    X = pd.DataFrame(
        [[float(body[feature]) for feature in FEATURES]],
        columns=FEATURES,
    )

    # ------------------------------------------------
    # MACHINE-LEARNING PREDICTION
    # ------------------------------------------------
    model_prediction = int(model.predict(X)[0])
    probability_severe = float(model.predict_proba(X)[0][1])

    model_label = "Severe Malaria" if model_prediction == 1 else "Not Severe Malaria"

    # ------------------------------------------------
    # CLINICAL SYMPTOM STATE
    # ------------------------------------------------
    active_symptoms = get_active_features(body, SYMPTOM_FEATURES)
    active_critical_symptoms = get_active_features(body, CRITICAL_SYMPTOMS)

    active_symptom_count = len(active_symptoms)
    critical_flags = len(active_critical_symptoms)

    # ------------------------------------------------
    # HYBRID RISK + FINAL DECISION
    # ------------------------------------------------
    severity_risk, risk_basis = calculate_severity_risk(
        probability_severe=probability_severe,
        active_symptom_count=active_symptom_count,
        critical_flags=critical_flags,
    )

    final_prediction = build_final_decision(
        model_prediction=model_prediction,
        severity_risk=severity_risk,
        risk_basis=risk_basis,
    )

    final_label = "Severe Malaria" if final_prediction == 1 else "Not Severe Malaria"

    prediction_certainty = calculate_prediction_certainty(
        probability_severe=probability_severe,
        critical_flags=critical_flags,
        risk_basis=risk_basis,
    )

    # ------------------------------------------------
    # SHAP EXPLAINABILITY
    # ------------------------------------------------
    shap_contributors = build_shap_contributors(X)
    top_contributors = shap_contributors[:7]

    # ------------------------------------------------
    # HUMAN-READABLE EXPLANATION LAYER
    # ------------------------------------------------
    explanation_summary = build_explanation_summary(
        risk_basis=risk_basis,
        active_symptoms=active_symptoms,
        active_critical_symptoms=active_critical_symptoms,
        top_contributors=top_contributors,
        body=body,
    )

    clinical_summary = build_clinical_summary(active_critical_symptoms)

    hybrid_reasoning = build_hybrid_reasoning(
        probability_severe=probability_severe,
        final_label=final_label,
        risk_basis=risk_basis,
        critical_flags=critical_flags,
        active_critical_symptoms=active_critical_symptoms,
        active_symptom_count=active_symptom_count,
        severity_risk=severity_risk,
    )

    model_warning = build_model_warning(top_contributors)

    # ------------------------------------------------
    # AUDIT-FRIENDLY API RESPONSE
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
        "hybrid_reasoning": hybrid_reasoning,
        "model_warning": model_warning,
    }
