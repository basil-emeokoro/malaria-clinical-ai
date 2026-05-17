from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np

# ============================================================
# LOAD MODEL + FEATURES
# ============================================================

model = joblib.load("model/model_v3.joblib")

FEATURES = joblib.load(
    "model/features_v3.joblib"
)

# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Malaria Severity Prediction API",
    version="5.0"
)

# ============================================================
# INPUT SCHEMA
# ============================================================

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


# ============================================================
# ROOT ROUTE
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Malaria Severity Prediction API Running"
    }

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/predict")
def predict(data: PatientData):

    # --------------------------------------------------------
    # Convert input to dataframe
    # --------------------------------------------------------

    body = data.dict()

    input_df = pd.DataFrame([body])

    input_df = input_df[FEATURES]

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    prediction = int(
        model.predict(input_df)[0]
    )

    # --------------------------------------------------------
    # CLINICAL OVERRIDE ENGINE
    # --------------------------------------------------------

    critical_flags = 0

    critical_symptoms = [
        "convulsion",
        "hypoglycemia",
        "prostration",
        "hyperpyrexia",
        "jaundice",
        "coca_cola_urine"
    ]

    for symptom in critical_symptoms:

        if body.get(symptom, 0) == 1:
            critical_flags += 1

    # --------------------------------------------------------
    # Risk escalation logic
    # --------------------------------------------------------

    if critical_flags >= 3:

        severity_risk = "HIGH"

    elif probability >= 0.55:

        severity_risk = "HIGH"

    elif probability >= 0.35:

        severity_risk = "MEDIUM"

    else:

        severity_risk = "LOW"

    # --------------------------------------------------------
    # Final diagnosis
    # --------------------------------------------------------

    if (
        prediction == 1
        or severity_risk == "HIGH"
    ):

        diagnosis = "Severe Malaria"

    else:

        diagnosis = "Not Severe Malaria"

    # --------------------------------------------------------
    # SHAP EXPLAINABILITY
    # --------------------------------------------------------

    shap_values = explainer.shap_values(input_df)

    severe_class_shap = shap_values[1][0]

    feature_impacts = []

    for feature, value in zip(
        FEATURES,
        severe_class_shap
    ):

        feature_impacts.append({
            "feature": feature,
            "impact": round(float(value), 4)
        })

    # --------------------------------------------------------
    # Sort by absolute impact
    # --------------------------------------------------------

    feature_impacts = sorted(
        feature_impacts,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    top_features = feature_impacts[:5]

    # --------------------------------------------------------
    # API RESPONSE
    # --------------------------------------------------------

    return {

        "diagnosis": diagnosis,

        "prediction": prediction,

        "severity_probability": round(
            probability * 100,
            2
        ),

        "risk_level": severity_risk,

        "critical_flags": critical_flags,

        "top_features": top_features
    }