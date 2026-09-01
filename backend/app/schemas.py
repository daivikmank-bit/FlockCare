"""Pydantic request and response schemas for the FlockCare API."""

from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    risk_score: float = Field(..., description="Screening risk score percentage [0, 100]")
    risk_level: str = Field(..., description="Risk tier: 'low' | 'moderate' | 'high'")
    message: str = Field(..., description="Clinical interpretation and flock management recommendation")
    disclaimer: str = Field(..., description="Medical / veterinary disclaimer")
    windows_analyzed: int = Field(..., description="Number of 5-second acoustic windows processed")
    status: str = Field(default="calibrated", description="'calibrated' if acoustic conditions match training, 'out_of_range' if anomalous")
    warning: Optional[str] = Field(default=None, description="Optional warning if audio conditions are out of calibrated range")
    ood_score: Optional[float] = Field(default=None, description="Mahalanobis distance to calibrated reference distribution")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Server health status")
