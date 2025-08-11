from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, SQLModel, create_engine, select

from .ml import (
    pca_2d,
    predict_kmeans,
    predict_propensity,
    train_kmeans,
    train_propensity,
)
from .models import (
    Customer,
    IngestResponse,
    PCAPoint,
    PredictRequest,
    PredictResponse,
    PropensityPredictRequest,
    PropensityPredictResponse,
    PropensityTrainRequest,
    PropensityTrainResponse,
    SegmentCount,
    SegmentStats,
    SegmentsSummaryResponse,
    Campaign,
    CampaignCreate,
    CampaignOut,
    AssignRequest,
    AssignResponse,
    Assignment,
    Outcome,
    OutcomeCreate,
    CampaignReport,
    CampaignVariantReport,
    TrainRequest,
    TrainResponse,
)
from .storage import load_model, save_model


DB_PATH = Path("data/db.sqlite")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


app = FastAPI(title="Precision Marketing Prototype")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.post("/ingest", response_model=IngestResponse)
def ingest(csv_file: UploadFile = File(...), session: Session = Depends(get_session)):
    try:
        df = pd.read_csv(csv_file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    required_cols = {"recency", "frequency", "monetary"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain columns: {sorted(required_cols)}",
        )

    if "avg_order" not in df.columns:
        # Derive if absent
        df["avg_order"] = (df["monetary"] / df["frequency"]).replace([pd.NA, pd.NaT], 0).fillna(0)

    # Standardize schema
    df = df[[
        c for c in ["customer_id", "recency", "frequency", "monetary", "avg_order"] if c in df.columns
    ]]

    rows = 0
    for _, row in df.iterrows():
        customer = Customer(
            customer_id=int(row["customer_id"]) if "customer_id" in df.columns else None,
            recency=float(row["recency"]),
            frequency=float(row["frequency"]),
            monetary=float(row["monetary"]),
            avg_order=float(row["avg_order"]),
        )
        session.add(customer)
        rows += 1
    session.commit()

    return IngestResponse(status="ok", rows=rows)


@app.post("/train/segment", response_model=TrainResponse)
def train_segment(payload: TrainRequest, session: Session = Depends(get_session)):
    features = payload.features
    if not features:
        raise HTTPException(status_code=400, detail="features list must not be empty")

    customers = session.exec(select(Customer)).all()
    if not customers:
        raise HTTPException(status_code=400, detail="No customers ingested")

    df = pd.DataFrame([c.dict() for c in customers])
    # Remove db primary key for training
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    model, metrics = train_kmeans(df, features, payload.n_clusters)
    save_model("kmeans.joblib", model)

    return TrainResponse(
        n_clusters=payload.n_clusters,
        features=features,
        inertia=float(metrics["inertia"]),
        centroids=metrics["centroids"],
    )


@app.post("/predict/segment", response_model=PredictResponse)
def predict_segment(payload: PredictRequest):
    try:
        model = load_model("kmeans.joblib")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained")

    # Infer features saved during training (fallback to default order)
    features = getattr(model, "feature_names_", ["recency", "frequency", "monetary"])
    missing = [f for f in features if f not in payload.customer]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    seg = predict_kmeans(model, payload.customer, features)
    return PredictResponse(segment=seg, prob_like=None)


@app.get("/segments/summary", response_model=SegmentsSummaryResponse)
def segments_summary(session: Session = Depends(get_session)):
    customers = session.exec(select(Customer)).all()
    if not customers:
        return SegmentsSummaryResponse(trained=False, counts=[], stats=[], pca=None)

    df = pd.DataFrame([c.dict() for c in customers])
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Attempt to load a model for assignments
    trained = True
    try:
        model = load_model("kmeans.joblib")
    except FileNotFoundError:
        trained = False
        model = None

    features = ["recency", "frequency", "monetary"]
    segments = None
    pca_points: List[PCAPoint] | None = None
    if trained and model is not None:
        # Predict segments
        try:
            segments = model.predict(df[features].astype(float))
        except Exception:
            segments = None

        # PCA for visualization
        try:
            pca_df = pca_2d(model, df, features)
            if segments is not None:
                pca_df["segment"] = segments
            else:
                pca_df["segment"] = -1
            pca_points = [PCAPoint(x=float(r.x), y=float(r.y), segment=int(r.segment)) for r in pca_df.itertuples(index=False)]
        except Exception:
            pca_points = None

    # Compute counts/stats
    if segments is not None:
        df["segment"] = segments
    else:
        df["segment"] = -1

    counts = (
        df.groupby("segment").size().reset_index(name="count").sort_values("segment")
    )
    stats = df.groupby("segment").agg(
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
        mean_avg_order=("avg_order", "mean"),
    ).reset_index()

    counts_models = [SegmentCount(segment=int(r.segment), count=int(r["count"])) for _, r in counts.iterrows()]
    stats_models = [
        SegmentStats(
            segment=int(r.segment),
            mean_recency=float(r.mean_recency),
            mean_frequency=float(r.mean_frequency),
            mean_monetary=float(r.mean_monetary),
            mean_avg_order=float(r.mean_avg_order),
        )
        for _, r in stats.iterrows()
    ]

    return SegmentsSummaryResponse(
        trained=trained,
        counts=counts_models,
        stats=stats_models,
        pca=pca_points,
    )


# --------- Propensity (Logistic Regression) ---------


@app.post("/train/propensity", response_model=PropensityTrainResponse)
def train_propensity_endpoint(payload: PropensityTrainRequest, session: Session = Depends(get_session)):
    customers = session.exec(select(Customer)).all()
    if not customers:
        raise HTTPException(status_code=400, detail="No customers ingested")

    df = pd.DataFrame([c.dict() for c in customers])
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Create labels based on strategy
    if payload.label_strategy == "median_monetary":
        thr = float(df["monetary"].median())
        y = (df["monetary"] >= thr).astype(int)
    elif payload.label_strategy == "median_frequency":
        thr = float(df["frequency"].median())
        y = (df["frequency"] >= thr).astype(int)
    elif payload.label_strategy == "custom_threshold":
        thr = payload.custom_threshold
        if thr is None:
            raise HTTPException(status_code=400, detail="custom_threshold required")
        y = (df["monetary"] >= float(thr)).astype(int)
    else:
        raise HTTPException(status_code=400, detail="Unknown label_strategy")

    model, info = train_propensity(df, payload.features, y)
    save_model("propensity.joblib", model)
    return PropensityTrainResponse(features=payload.features, coef=info.get("coef"), intercept=info.get("intercept"))


@app.post("/predict/propensity", response_model=PropensityPredictResponse)
def predict_propensity_endpoint(payload: PropensityPredictRequest):
    try:
        model = load_model("propensity.joblib")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Propensity model not trained")
    features = getattr(model, "feature_names_", ["recency", "frequency", "monetary", "avg_order"])
    proba, klass = predict_propensity(model, payload.customer, features)
    return PropensityPredictResponse(probability=proba, klass=klass)


# --------- A/B Experiments ---------


def _assign_variant_deterministic(campaign_id: int, customer_id: int, variants: List[str]) -> str:
    import hashlib

    h = hashlib.sha256(f"{campaign_id}:{customer_id}".encode()).hexdigest()
    r = int(h[:8], 16) / 0xFFFFFFFF
    idx = int(r * len(variants)) % len(variants)
    return variants[idx]


@app.post("/campaigns", response_model=CampaignOut)
def create_campaign(payload: CampaignCreate, session: Session = Depends(get_session)):
    camp = Campaign(name=payload.name, start_ts=payload.start_ts or pd.Timestamp.utcnow().to_pydatetime(), end_ts=payload.end_ts)
    session.add(camp)
    session.commit()
    session.refresh(camp)
    return CampaignOut(id=camp.id, name=camp.name, start_ts=camp.start_ts, end_ts=camp.end_ts)


@app.post("/assign", response_model=AssignResponse)
def assign_variant(payload: AssignRequest, session: Session = Depends(get_session)):
    variant = _assign_variant_deterministic(payload.campaign_id, payload.customer_id, payload.variants)
    rec = Assignment(campaign_id=payload.campaign_id, customer_id=payload.customer_id, variant=variant)
    session.add(rec)
    session.commit()
    return AssignResponse(campaign_id=payload.campaign_id, customer_id=payload.customer_id, variant=variant)


@app.post("/outcomes")
def report_outcome(payload: OutcomeCreate, session: Session = Depends(get_session)):
    rec = Outcome(
        campaign_id=payload.campaign_id,
        customer_id=payload.customer_id,
        converted=payload.converted,
        revenue=payload.revenue,
    )
    session.add(rec)
    session.commit()
    return {"status": "ok"}


@app.get("/campaigns/{campaign_id}/report", response_model=CampaignReport)
def campaign_report(campaign_id: int, session: Session = Depends(get_session)):
    import numpy as np

    assignments = session.exec(select(Assignment).where(Assignment.campaign_id == campaign_id)).all()
    outcomes = session.exec(select(Outcome).where(Outcome.campaign_id == campaign_id)).all()
    if not assignments:
        raise HTTPException(status_code=404, detail="No assignments found")

    # Join in-memory
    df_a = pd.DataFrame([a.dict() for a in assignments])
    df_o = pd.DataFrame([o.dict() for o in outcomes]) if outcomes else pd.DataFrame(columns=["customer_id", "converted", "revenue"]).assign(converted=False, revenue=0.0)
    df = df_a.merge(df_o[["customer_id", "converted", "revenue"]], on="customer_id", how="left").fillna({"converted": False, "revenue": 0.0})

    reports: List[CampaignVariantReport] = []
    for variant, g in df.groupby("variant"):
        assigned = int(len(g))
        conversions = int(g["converted"].sum())
        cr = float(conversions / assigned) if assigned else 0.0
        avg_rev = float(np.mean(g["revenue"])) if assigned else 0.0
        reports.append(CampaignVariantReport(variant=str(variant), assigned=assigned, conversions=conversions, conversion_rate=cr, avg_revenue=avg_rev))

    return CampaignReport(campaign_id=campaign_id, variants=reports)


