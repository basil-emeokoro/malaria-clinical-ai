"""
app.py — Malaria Severity Prediction API (V3.2)
------------------------------------------------
Hybrid Clinical ML API

Features:
✅ Stable RandomForest inference
✅ No scaling drift
✅ Clean feature schema
✅ Explainability-ready
✅ SHAP-ready architecture
✅ Clinical override engine
✅ Payload debugging
"""

import logging
import joblib
import pandas as pd

from flask import Flask, request, jsonify
from datetime import datetime, UTC

# ------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)

# ------------------------------------------------
# FLASK APP INITIALIZATION
# ------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------
# LOAD TRAINED MODEL + FEATURES
# ------------------------------------------------
model = joblib.load("model/model_v3.joblib")

FEATURES = joblib.load(
    "model/features_v3.joblib"
)

# ------------------------------------------------
# LIVE API STATISTICS
# ------------------------------------------------
stats = {
    "total_requests": 0,
    "severe_predicted": 0,
    "not_severe_predicted": 0,
    "errors": 0,
    "started_at": datetime.now(UTC).isoformat()
}

# ------------------------------------------------
# HEALTH CHECK ENDPOINT
# ------------------------------------------------
@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": True
    }), 200


# ------------------------------------------------
# MODEL INFO ENDPOINT
# ------------------------------------------------
@app.route("/info")
def info():

    return jsonify({
        "model": "Malaria Severity Classifier",
        "version": "3.2.0",
        "description": (
            "Hybrid malaria severity prediction API using "
            "machine learning probability plus clinical override rules."
        ),
        "features": FEATURES,
        "target": "severe_malaria (0 = not severe, 1 = severe)",
        "endpoint": "POST /predict"
    }), 200


# ------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Track request count
    stats["total_requests"] += 1

    # Parse incoming JSON
    body = request.get_json(silent=True)

    # Debug raw input
    print("\n================================================")
    print("[RAW INPUT DATA]")
    print(body)
    print("================================================")

    if body is None:
        stats["errors"] += 1
        return jsonify({
            "error": "Invalid JSON request."
        }), 400

    # Validate required features
    missing = [
        f for f in FEATURES
        if f not in body
    ]

    if missing:
        stats["errors"] += 1
        return jsonify({
            "error": "Missing features.",
            "missing": missing,
            "required": FEATURES
        }), 422

    # Convert request into DataFrame
    try:
        X = pd.DataFrame(
            [[float(body[f]) for f in FEATURES]],
            columns=FEATURES
        )

        # Debug model input
        print("\n[MODEL INPUT]")
        print(X)
        print("================================================\n")

    except Exception as e:
        stats["errors"] += 1
        return jsonify({
            "error": str(e)
        }), 422

    # ------------------------------------------------
    # RUN ML MODEL PREDICTION
    # ------------------------------------------------
    prediction = int(
        model.predict(X)[0]
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    # ------------------------------------------------
    # CLINICAL OVERRIDE ENGINE
    # ------------------------------------------------
    # Rationale:
    # The dataset is small and the ML model is conservative.
    # Therefore, high-risk clinical indicators are used as an
    # escalation safeguard, similar to real clinical decision support.
    # ------------------------------------------------

    critical_symptoms = [
        "convulsion",
        "hypoglycemia",
        "prostration",
        "hyperpyrexia",
        "jaundice",
        "coca_cola_urine"
    ]

    critical_flags = 0
    active_critical_symptoms = []

    for symptom in critical_symptoms:
        if int(body.get(symptom, 0)) == 1:
            critical_flags += 1
            active_critical_symptoms.append(symptom)

    # Decide risk level using hybrid logic:
    # 1. Clinical override first
    # 2. Then ML probability thresholds
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

    # Human-readable diagnosis label still follows the ML class prediction
    label = (
        "Severe Malaria"
        if prediction == 1
        else "Not Severe Malaria"
    )

    # Update prediction statistics
    if prediction == 1:
        stats["severe_predicted"] += 1
    else:
        stats["not_severe_predicted"] += 1

    # ------------------------------------------------
    # EXPLAINABILITY SECTION
    # ------------------------------------------------
    top_contributors = []

    try:
        if hasattr(model, "feature_importances_"):

            importances = pd.Series(
                model.feature_importances_,
                index=FEATURES
            ).sort_values(ascending=False)

            top_contributors = [
                {
                    "feature": f,
                    "importance": round(
                        float(importances[f]),
                        4
                    )
                }
                for f in importances.head(5).index
            ]

    except Exception as e:
        logger.warning(
            f"Explainability failed: {e}"
        )

    # Log prediction
    logger.info(
        f"Prediction → {label} "
        f"(prob={probability:.3f}, risk={severity_risk}, basis={risk_basis})"
    )

    # ------------------------------------------------
    # FINAL API RESPONSE
    # ------------------------------------------------
    return jsonify({
        "prediction": prediction,
        "label": label,
        "probability_severe": round(probability, 4),
        "severity_risk": severity_risk,
        "risk_basis": risk_basis,
        "critical_flags": critical_flags,
        "active_critical_symptoms": active_critical_symptoms,
        "top_contributors": top_contributors
    }), 200


# ------------------------------------------------
# LIVE STATS ENDPOINT
# ------------------------------------------------
@app.route("/stats")
def get_stats():

    return jsonify(stats), 200


# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":

    logger.info(
        "Starting Malaria Severity "
        "Prediction API on port 5000 ..."
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )