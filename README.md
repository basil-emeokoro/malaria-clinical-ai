Malaria Clinical AI — Explainable Severity Prediction System
Overview
Malaria Clinical AI is an Explainable Artificial Intelligence (XAI)-powered clinical decision support prototype designed to predict malaria severity risk using patient symptoms and demographic indicators.
The system combines:


Machine Learning


Explainable AI (XAI)


Flask REST API deployment


Streamlit clinical dashboard


Real-time prediction analytics


Responsible AI debugging workflow


This project was developed as part of the ADA Global Data Science learning and deployment portfolio.

Project Goals
The system aims to:


Predict severe malaria risk from patient symptom profiles


Demonstrate Explainable AI integration in healthcare ML


Provide interpretable clinical predictions


Explore trustworthy AI deployment practices


Simulate real-world ML engineering workflow


Support research in responsible healthcare AI systems



Current Features
Machine Learning Pipeline


Random Forest-based severity prediction


Structured feature engineering


Clinical symptom encoding


Real-time inference pipeline


Probability-based risk estimation



Explainable AI (XAI)
Current explainability includes:


Feature importance extraction


Prediction transparency


Top contributing factors


Interactive contribution charts


Explainability-aware UI


Planned upgrades:


SHAP-based patient-specific explanations


Local explanation analysis


Explainability governance auditing



Clinical Dashboard
The Streamlit dashboard provides:


Patient symptom input interface


Risk classification


Probability estimation


Visual severity indicators


Explainability charts


Clinical recommendation summaries



Technology Stack
ltc1q7mhxnw82zyzkjvdtv57geqjjsw0mhgrvq6nx83 FrameworkScikit-learnAPI BackendFlaskFrontend UIStreamlitExplainabilitySHAP (planned integration)Data HandlingPandas / NumPyDeploymentDocker-readyVersion ControlGit + GitHub

Project Structure
malaria-clinical-ai/│├── data/│   └── Malaria-Data.csv│├── model/│   ├── model.joblib│   ├── model_v2.joblib│   ├── features.joblib│   ├── features_v2.joblib│   └── scaler.joblib│├── src/│   ├── app.py│   ├── ui.py│   ├── train.py│   ├── train_v2.py│   ├── diagnose_data.py│   └── check_labels.py│├── Dockerfile├── requirements.txt└── README.md

API Endpoints
Health Check
GET /health

Model Information
GET /info

Prediction Endpoint
POST /predict
Example Request
{  "age": 25,  "sex": 0,  "fever": 1,  "cold": 1,  "rigor": 1,  "fatigue": 1,  "headace": 1,  "bitter_tongue": 1,  "vomitting": 1,  "diarrhea": 1,  "Convulsion": 1,  "Anemia": 1,  "jundice": 1,  "cocacola_urine": 1,  "hypoglycemia": 1,  "prostraction": 1,  "hyperpyrexia": 1}

Example Response
{  "prediction": 1,  "label": "Severe Malaria",  "probability_severe": 0.5079,  "severity_risk": "HIGH",  "top_contributors": [    {      "feature": "age",      "importance": 0.2687    },    {      "feature": "headace",      "importance": 0.0613    }  ]}

Model Evaluation
Current evaluation metrics:
MetricCurrent StatusAccuracyModerateROC-AUCImprovingExplainabilityIntegratedClinical ConsistencyImprovingDeployment StabilityStable

Explainability & Responsible AI
One of the major goals of this project is to demonstrate how Explainable AI can help detect unsafe or inconsistent ML behavior before deployment.
During development, explainability analysis exposed:


unstable severity predictions


inference inconsistencies


preprocessing drift


weak feature sensitivity


This enabled redesign of the prediction architecture into a more stable and transparent clinical AI pipeline.

Current Development Roadmap
Phase 1 — Completed


Baseline ML model


Flask deployment


Streamlit dashboard


Explainability charts


GitHub deployment


Prediction auditing


Clinical debugging workflow



Phase 2 — In Progress
Goals:


Improve clinical consistency


Improve severe case sensitivity


Improve ROC-AUC


Improve explainability realism


Planned upgrades:


Balanced Random Forest


SHAP integration


Improved calibration


Better threshold handling


Enhanced model validation



Research & Educational Value
This project demonstrates concepts in:


Explainable AI (XAI)


Responsible AI


Clinical Decision Support Systems


Healthcare ML Governance


AI Auditing


Deployment Engineering


Human-Centered AI



Important Disclaimer
This system is a research and educational prototype.
It is NOT intended for real clinical deployment or medical diagnosis.
Predictions should not be used for real-world medical decision-making.

Future Directions
Potential future enhancements include:


SHAP waterfall visualizations


XGBoost comparison


LightGBM benchmarking


Calibration curves


Fairness evaluation


Bias auditing


Cloud deployment


Real-time monitoring


Multi-disease prediction expansion



Author
Basil Emeokoro co-authoring with ADA Global Academy.
Research interests include:


Explainable AI


Educational Assessment Systems


Responsible AI


Clinical AI Governance


AI Accountability


Digital Assessment Infrastructure


GitHub:
https://github.com/basil-emeokoro

License
This repository is intended for educational and research purposes.
