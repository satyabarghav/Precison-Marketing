from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, SQLModel


class Customer(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    customer_id: Optional[int] = SQLField(index=True, default=None)
    recency: float
    frequency: float
    monetary: float
    avg_order: float


# ---- Request/Response Models ----


class IngestResponse(BaseModel):
    status: str
    rows: int


class TrainRequest(BaseModel):
    features: List[str] = Field(default_factory=lambda: ["recency", "frequency", "monetary"])
    n_clusters: int = 4


class TrainResponse(BaseModel):
    status: str = "ok"
    n_clusters: int
    features: List[str]
    inertia: float
    centroids: List[List[float]]


class PredictRequest(BaseModel):
    customer: Dict[str, float]


class PredictResponse(BaseModel):
    segment: int
    prob_like: Optional[float] = None


class SegmentCount(BaseModel):
    segment: int
    count: int


class SegmentStats(BaseModel):
    segment: int
    mean_recency: Optional[float] = None
    mean_frequency: Optional[float] = None
    mean_monetary: Optional[float] = None
    mean_avg_order: Optional[float] = None


class PCAPoint(BaseModel):
    x: float
    y: float
    segment: int


class SegmentsSummaryResponse(BaseModel):
    trained: bool
    counts: List[SegmentCount]
    stats: List[SegmentStats]
    pca: Optional[List[PCAPoint]] = None


