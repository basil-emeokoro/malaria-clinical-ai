# Deployment Guide

This project deploys as two services:

- FastAPI backend on Render
- Streamlit frontend on Streamlit Community Cloud

The active production stack is `src/app.py`, `src/ui.py`, `model/model_v3.joblib`, `model/features_v3.joblib`, and `src/train_v3.py`.

## Local Run

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.app:app --reload
```

In another terminal:

```powershell
streamlit run src/ui.py
```

Local API checks:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/info
```

## Docker Local Run

```powershell
docker build -t malaria-clinical-ai .
docker run -p 8000:8000 malaria-clinical-ai
```

If local port `8000` is already occupied, keep container port `8000` and change only the host port:

```powershell
docker run -p 8010:8000 malaria-clinical-ai
```

## Render Backend Deployment

1. Push the repository to GitHub.
2. Create a Render Web Service from the GitHub repository.
3. Select Docker deployment.
4. Use the repository root as the Docker build context.
5. Do not change the container internal port from `8000`.
6. Render provides the `PORT` environment variable automatically.

Render settings:

```text
Runtime: Docker
Build Command: Dockerfile managed by Render
Start Command: Dockerfile CMD
Environment Variables: none required for backend; Render supplies PORT automatically
```

The Docker command is:

```text
uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}
```

Verify:

```text
https://YOUR-RENDER-SERVICE.onrender.com/health
https://YOUR-RENDER-SERVICE.onrender.com/info
```

## Streamlit Community Cloud Frontend

1. Create a Streamlit Community Cloud app from the same GitHub repository.
2. Set the app entrypoint to `streamlit_app.py`.
3. Confirm dependencies install from `requirements.txt`.
4. Confirm the dashboard loads.

## Update API_URL After Render Deployment

After Render gives the public backend URL, update this line in `src/ui.py`:

```python
API_URL = "http://127.0.0.1:8000/predict"
```

Change it to:

```python
API_URL = "https://YOUR-RENDER-SERVICE.onrender.com/predict"
```

Commit and redeploy Streamlit after this change.

## Final Validation

```powershell
python -m py_compile src/app.py
python -m py_compile src/ui.py
python -m py_compile src/train_v3.py
```

Confirm these model files exist:

```text
model/model_v3.joblib
model/features_v3.joblib
```
