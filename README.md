# Precision Marketing Prototype

Fast prototype for ingesting customer CSVs, training KMeans segments on RFM-like features, and visualizing segments via Streamlit.

## Tech stack
- Backend: FastAPI, Uvicorn, Pandas, scikit-learn, joblib, SQLModel, pydantic
- Frontend: Streamlit, requests, Altair

## Repo layout
```
precision-marketing-proto/
├─ backend/
│  ├─ main.py             # FastAPI app
│  ├─ models.py           # pydantic & DB models
│  ├─ ml.py               # dataset gen, training, predict helpers
│  ├─ storage.py          # save/load models, DB helpers
│  └─ requirements.txt
├─ frontend/
│  ├─ app.py              # Streamlit app
│  └─ requirements.txt
├─ sample_data/
│  └─ synthetic_customers.csv
└─ README.md
```

## Quickstart

1) Create virtual envs and install dependencies
```bash
cd precision-marketing-proto
python -m venv .venv && . .venv/Scripts/activate  # On Windows PowerShell: . .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

2) Generate sample data
```bash
python -c "from backend.ml import generate_synthetic_data; generate_synthetic_data()"
```

3) Run backend
```bash
uvicorn backend.main:app --reload
```

4) Run frontend (in a second terminal)
```bash
streamlit run frontend/app.py
```

5) Demo workflow
- Upload `sample_data/synthetic_customers.csv` in the Ingest tab
- Train segments (k=4)
- View PCA 2D scatter colored by segment
- Predict a single customer segment

## API endpoints
- POST `/ingest` (multipart form with `csv_file`)
- POST `/train/segment` JSON: `{ "features": ["recency","frequency","monetary"], "n_clusters": 4 }`
- POST `/predict/segment` JSON: `{ "customer": {"recency":10,"frequency":5,"monetary":250} }`
- GET `/segments/summary`


