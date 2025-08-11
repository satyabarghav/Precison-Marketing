from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def generate_synthetic_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    import numpy as _np
    _np.random.seed(seed)
    frequency = _np.random.poisson(3, n) + 1
    monetary = _np.round(_np.random.gamma(2.0, 50.0, n) + frequency * 20, 2)
    recency = _np.random.exponential(30, n).astype(int)
    avg_order = _np.round(monetary / frequency, 2)
    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_order": avg_order,
        }
    )
    out_dir = Path("sample_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "synthetic_customers.csv", index=False)
    return df


def build_pipeline(n_clusters: int) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=10, random_state=42)),
        ]
    )
    return pipeline


def train_kmeans(df: pd.DataFrame, features: List[str], n_clusters: int) -> Tuple[Pipeline, Dict]:
    data = df[features].astype(float).copy()
    pipeline = build_pipeline(n_clusters)
    pipeline.fit(data)
    # Persist training features on the pipeline for inference
    setattr(pipeline, "feature_names_", list(features))

    kmeans: KMeans = pipeline.named_steps["kmeans"]
    scaler: StandardScaler = pipeline.named_steps["scaler"]

    # Report centroids in original feature space for interpretability
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    metrics = {
        "inertia": float(kmeans.inertia_),
        "centroids": centroids_original.tolist(),
    }
    return pipeline, metrics


def predict_kmeans(model: Pipeline, row: Dict[str, float], features: List[str]) -> int:
    X = pd.DataFrame([row], columns=features).astype(float)
    segment = int(model.predict(X)[0])
    return segment


def pca_2d(model: Pipeline, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    data = df[features].astype(float)
    scaler: StandardScaler = model.named_steps["scaler"]
    transformed = scaler.transform(data)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(transformed)
    out = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    return out


def train_propensity(
    df: pd.DataFrame,
    features: List[str],
    labels: pd.Series,
) -> Tuple[Pipeline, Dict]:
    X = df[features].astype(float)
    y = labels.astype(int)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    setattr(pipeline, "feature_names_", list(features))
    clf: LogisticRegression = pipeline.named_steps["clf"]
    coef = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])
    return pipeline, {"coef": coef, "intercept": intercept}


def predict_propensity(model: Pipeline, row: Dict[str, float], features: List[str]) -> Tuple[float, int]:
    X = pd.DataFrame([row], columns=features).astype(float)
    proba = float(model.predict_proba(X)[0, 1])
    klass = int(model.predict(X)[0])
    return proba, klass


