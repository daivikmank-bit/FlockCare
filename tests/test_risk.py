"""Unit tests for clinical risk score and label mapping."""

import pytest
from ml.models.risk import to_risk_label


def test_to_risk_label_low():
    res = to_risk_label(0.123)
    assert res["risk_score"] == 12.3
    assert res["risk_level"] == "low"
    assert "healthy" in res["message"].lower()


def test_to_risk_label_moderate():
    res = to_risk_label(0.45)
    assert res["risk_score"] == 45.0
    assert res["risk_level"] == "moderate"
    assert "stress" in res["message"].lower() or "monitor" in res["message"].lower()


def test_to_risk_label_high():
    res = to_risk_label(0.789)
    assert res["risk_score"] == 78.9
    assert res["risk_level"] == "high"
    assert "isolate" in res["message"].lower() or "veterinarian" in res["message"].lower()


def test_to_risk_label_boundary_conditions():
    # Exactly 0.0
    r0 = to_risk_label(0.0)
    assert r0["risk_score"] == 0.0
    assert r0["risk_level"] == "low"

    # Exactly 0.399 -> 39.9 -> low
    r_sub40 = to_risk_label(0.399)
    assert r_sub40["risk_score"] == 39.9
    assert r_sub40["risk_level"] == "low"

    # Exactly 0.40 -> 40.0 -> moderate
    r40 = to_risk_label(0.40)
    assert r40["risk_score"] == 40.0
    assert r40["risk_level"] == "moderate"

    # Exactly 0.699 -> 69.9 -> moderate
    r_sub70 = to_risk_label(0.699)
    assert r_sub70["risk_score"] == 69.9
    assert r_sub70["risk_level"] == "moderate"

    # Exactly 0.70 -> 70.0 -> high
    r70 = to_risk_label(0.70)
    assert r70["risk_score"] == 70.0
    assert r70["risk_level"] == "high"

    # Exactly 1.0 -> 100.0 -> high
    r100 = to_risk_label(1.0)
    assert r100["risk_score"] == 100.0
    assert r100["risk_level"] == "high"


def test_to_risk_label_out_of_bounds_clamping():
    r_neg = to_risk_label(-0.5)
    assert r_neg["risk_score"] == 0.0
    assert r_neg["risk_level"] == "low"

    r_over = to_risk_label(1.5)
    assert r_over["risk_score"] == 100.0
    assert r_over["risk_level"] == "high"
