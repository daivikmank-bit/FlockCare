"""Risk score calculation and farmer-facing risk classification."""

from typing import Any, Dict


def to_risk_label(prob_elevated: float) -> Dict[str, Any]:
    """
    Convert model output probability for elevated respiratory sounds into clinical risk classification.
    
    Thresholds:
      - score >= 70%: 'high' risk -> Urgent veterinary guidance and flock isolation.
      - score >= 40%: 'moderate' risk -> Active monitoring over 24-48 hours.
      - score < 40%: 'low' risk -> Healthy flock sounds.
      
    Args:
        prob_elevated: Probability of elevated respiratory distress (0.0 to 1.0).
        
    Returns:
        Dictionary containing:
          - risk_score: Float percentage (0.0 - 100.0) rounded to 1 decimal place.
          - risk_level: 'high' | 'moderate' | 'low'.
          - message: Actionable guidance string for the poultry farmer.
    """
    # Ensure clamped float in range [0.0, 1.0]
    prob = float(max(0.0, min(1.0, float(prob_elevated))))
    score = round(prob * 100, 1)

    if score >= 70.0:
        level = "high"
        message = (
            "Elevated respiratory sounds detected — isolate the flock and consult a veterinarian."
        )
    elif score >= 40.0:
        level = "moderate"
        message = (
            "Some signs of respiratory stress. Monitor closely over the next 24–48 hours."
        )
    else:
        level = "low"
        message = "Flock sounds healthy. No signs of respiratory distress detected."

    return {
        "risk_score": score,
        "risk_level": level,
        "message": message,
    }
