from __future__ import annotations

from datetime import datetime
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


# ---- Propensity (Likelihood to Purchase) ----


class PropensityTrainRequest(BaseModel):
    features: List[str] = Field(default_factory=lambda: ["recency", "frequency", "monetary", "avg_order"])
    label_strategy: str = Field(
        default="median_monetary",
        description="How to create labels from current data: median_monetary | median_frequency | custom_threshold",
    )
    custom_threshold: Optional[float] = Field(default=None, description="Used when label_strategy=custom_threshold against 'monetary'")


class PropensityTrainResponse(BaseModel):
    status: str = "ok"
    features: List[str]
    coef: Optional[List[float]] = None
    intercept: Optional[float] = None


class PropensityPredictRequest(BaseModel):
    customer: Dict[str, float]


class PropensityPredictResponse(BaseModel):
    probability: float
    klass: int


# ---- Experiments (A/B) ----


class Campaign(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    name: str
    start_ts: datetime = SQLField(default_factory=datetime.utcnow)
    end_ts: Optional[datetime] = None


class Assignment(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    campaign_id: int = SQLField(index=True)
    customer_id: int = SQLField(index=True)
    variant: str = SQLField(index=True)
    assigned_ts: datetime = SQLField(default_factory=datetime.utcnow)
    sticky_hash: Optional[str] = None


class Outcome(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    campaign_id: int = SQLField(index=True)
    customer_id: int = SQLField(index=True)
    converted: bool
    revenue: float = 0.0
    observed_ts: datetime = SQLField(default_factory=datetime.utcnow)


class CampaignCreate(BaseModel):
    name: str
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None


class CampaignOut(BaseModel):
    id: int
    name: str
    start_ts: datetime
    end_ts: Optional[datetime] = None


class AssignRequest(BaseModel):
    campaign_id: int
    customer_id: int
    variants: List[str] = Field(default_factory=lambda: ["A", "B"])


class AssignResponse(BaseModel):
    campaign_id: int
    customer_id: int
    variant: str


class OutcomeCreate(BaseModel):
    campaign_id: int
    customer_id: int
    converted: bool
    revenue: float = 0.0


class CampaignVariantReport(BaseModel):
    variant: str
    assigned: int
    conversions: int
    conversion_rate: float
    avg_revenue: float


class CampaignReport(BaseModel):
    campaign_id: int
    variants: List[CampaignVariantReport]


