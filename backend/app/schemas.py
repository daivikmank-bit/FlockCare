"""Pydantic request and response schemas for the FlockCare API."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class WindowDetail(BaseModel):
    window_index: int = Field(..., description="0-indexed window number")
    start_sec: float = Field(..., description="Start timestamp in seconds")
    end_sec: float = Field(..., description="End timestamp in seconds")
    risk_score: float = Field(..., description="Window-level respiratory risk score [0, 100]")
    ood_score: float = Field(..., description="Mahalanobis distance for this window")
    is_ood: bool = Field(..., description="Whether this window exceeded the OOD threshold")
    spectrogram_image: str = Field(..., description="Base64 data URL of the log-mel spectrogram")
    heatmap_image: str = Field(..., description="Base64 data URL of the Grad-CAM saliency heatmap")
    biomarkers: Dict[str, float] = Field(..., description="Acoustic biomarkers for this window")


class FeatureImportanceItem(BaseModel):
    feature_name: str = Field(..., description="Name of the acoustic feature")
    value: str = Field(..., description="Measured value formatted as string")
    impact: float = Field(..., description="SHAP-style risk impact (+/- %)")
    direction: str = Field(..., description="'increases_risk' | 'decreases_risk'")
    clinical_significance: str = Field(..., description="Veterinary interpretation")


class DiseaseDifferentialItem(BaseModel):
    disease_id: str
    name: str
    pathogen: str
    likelihood: str = Field(..., description="'High' | 'Moderate' | 'Possible' | 'Low'")
    probability_pct: int
    acoustic_rationale: str
    is_notifiable: bool
    key_symptoms: List[str]
    biosecurity_actions: List[str]


class DiseaseDifferentialSummary(BaseModel):
    flock_clinical_status: str
    primary_concern: str
    differentials: List[DiseaseDifferentialItem]
    overall_biosecurity_advice: List[str]


class AnalyzeResponse(BaseModel):
    risk_score: float = Field(..., description="Screening risk score percentage [0, 100]")
    risk_level: str = Field(..., description="Risk tier: 'low' | 'moderate' | 'high'")
    message: str = Field(..., description="Clinical interpretation and flock management recommendation")
    disclaimer: str = Field(..., description="Medical / veterinary disclaimer")
    windows_analyzed: int = Field(..., description="Number of 5-second acoustic windows processed")
    status: str = Field(default="calibrated", description="'calibrated' if acoustic conditions match training, 'out_of_range' if anomalous")
    warning: Optional[str] = Field(default=None, description="Optional warning if audio conditions are out of calibrated range")
    ood_score: Optional[float] = Field(default=None, description="Mahalanobis distance to calibrated reference distribution")
    windows_detail: Optional[List[WindowDetail]] = Field(default=None, description="Per-window spectrograms, Grad-CAM heatmaps, and biomarkers")
    overall_biomarkers: Optional[Dict[str, float]] = Field(default=None, description="Flock-wide aggregated acoustic biomarkers")
    feature_importance: Optional[List[FeatureImportanceItem]] = Field(default=None, description="SHAP-style acoustic feature attribution")
    disease_differential: Optional[DiseaseDifferentialSummary] = Field(default=None, description="Expected avian disease differential diagnoses and checklist")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Server health status")
