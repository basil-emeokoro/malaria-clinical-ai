# Malaria Clinical AI - Hybrid Explainable Decision Support Prototype

Research prototype for safety-aware malaria severity decision support using a FastAPI backend, Streamlit frontend, RandomForest V3 model, clinical safety layer, and SHAP explainability.

## Current Architecture

- Backend: FastAPI application in `src/app.py`
- Frontend: Streamlit dashboard in `src/ui.py`
- Streamlit Cloud entrypoint: `streamlit_app.py`
- Model: RandomForest V3
- Active model artifacts: `model/model_v3.joblib`, `model/features_v3.joblib`
- Reproducibility trainer: `src/train_v3.py`
- Explainability: SHAP TreeExplainer
- Clinical safety layer: rule-based safety assessment layered on model probability

The active backend uses the V3 model artifacts. Do not switch deployment to V4, `pipeline.joblib`, or archived original Flask/training files unless explicitly retraining and revalidating.

## Disclaimer

This dashboard is for academic, research, and demonstration use only. It is not a medical device and must not be used as a final diagnosis. Always confirm with qualified clinical evaluation and laboratory testing.

## Local Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Run backend:

```powershell
uvicorn src.app:app --reload
```

Run frontend:

```powershell
streamlit run src/ui.py
```

## API Endpoints

- `GET /health`
- `GET /info`
- `POST /predict`

Example `/predict` payload:

```json
{
  "age": 25,
  "sex": 0,
  "fever": 1,
  "cold": 0,
  "rigor": 1,
  "fatigue": 1,
  "headache": 1,
  "bitter_tongue": 0,
  "vomiting": 0,
  "diarrhea": 1,
  "convulsion": 0,
  "anemia": 0,
  "jaundice": 0,
  "coca_cola_urine": 0,
  "hypoglycemia": 0,
  "prostration": 0,
  "hyperpyrexia": 0
}
```

## Docker Backend

Build:

```powershell
docker build -t malaria-clinical-ai .
```

Run:

```powershell
docker run -p 8000:8000 malaria-clinical-ai
```

If local port `8000` is occupied:

```powershell
docker run -p 8010:8000 malaria-clinical-ai
```

The container still exposes internal port `8000`, preserving Render compatibility.

## Render Backend Deployment

1. Push this repository to GitHub.
2. Create a Render Web Service from the GitHub repository.
3. Choose Docker deployment.
4. Use the repository root as the Docker build context.
5. Use these settings:

```text
Runtime: Docker
Build Command: Dockerfile managed by Render
Start Command: Dockerfile CMD
Environment Variables: none required for backend; Render supplies PORT automatically
```

6. Keep the Dockerfile command:

```text
uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}
```

7. Verify:

```text
https://YOUR-RENDER-SERVICE.onrender.com/health
https://YOUR-RENDER-SERVICE.onrender.com/info
```

## Streamlit Community Cloud

1. Create a Streamlit app from this GitHub repository.
2. Use `streamlit_app.py` as the entrypoint.
3. Streamlit installs dependencies from `requirements.txt`.
4. After Render provides a public backend URL, update `API_URL` in `src/ui.py`:

```python
API_URL = "https://YOUR-RENDER-SERVICE.onrender.com/predict"
```

5. Redeploy Streamlit and capture at least three prediction screenshots.

## Expected Model Files

```text
model/model_v3.joblib
model/features_v3.joblib
```

## Submission Checklist

- GitHub repository URL ready
- Render backend URL ready
- Streamlit frontend URL ready
- `/health`, `/info`, and `/predict` tested
- At least three prediction screenshots captured
- 2-page reflection completed
- Optional 5-minute demo video completed
- Final commit hash recorded
- Final tag recorded: `v1.0-capstone-ready`
