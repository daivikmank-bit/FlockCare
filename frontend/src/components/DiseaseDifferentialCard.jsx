import React, { useState } from "react";
import { AlertCircle, ChevronDown, ChevronUp, ShieldAlert, CheckSquare, Square, Stethoscope, AlertTriangle, ShieldCheck } from "lucide-react";

export default function DiseaseDifferentialCard({ diseaseDifferential, t }) {
  const [expandedId, setExpandedId] = useState(null);
  const [checkedSymptoms, setCheckedSymptoms] = useState({});

  if (!diseaseDifferential || !diseaseDifferential.differentials) return null;

  function toggleExpand(id) {
    setExpandedId(expandedId === id ? null : id);
  }

  function toggleSymptom(key) {
    setCheckedSymptoms((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  }

  const likelihoodBadgeStyles = {
    High: { bg: "rgba(239, 68, 68, 0.15)", border: "#ef4444", text: "#ef4444" },
    Moderate: { bg: "rgba(245, 158, 11, 0.15)", border: "#f59e0b", text: "#f59e0b" },
    Possible: { bg: "rgba(59, 130, 246, 0.15)", border: "#3b82f6", text: "#60a5fa" },
    Low: { bg: "rgba(16, 185, 129, 0.15)", border: "#10b981", text: "#34d399" },
  };

  return (
    <div className="disease-differential-box">
      <div className="differential-header">
        <div className="diff-header-left">
          <Stethoscope size={20} className="text-primary" />
          <div>
            <h3 className="diff-title">Expected Avian Disease Differential</h3>
            <p className="diff-subtitle">{diseaseDifferential.primary_concern}</p>
          </div>
        </div>
      </div>

      {/* Disease Cards List */}
      <div className="disease-cards-list">
        {diseaseDifferential.differentials.map((disease) => {
          const isExpanded = expandedId === disease.disease_id;
          const badgeStyle = likelihoodBadgeStyles[disease.likelihood] || likelihoodBadgeStyles.Low;

          return (
            <div
              key={disease.disease_id}
              className={`disease-card ${disease.is_notifiable ? "notifiable-border" : ""} ${isExpanded ? "expanded" : ""}`}
            >
              {/* Header summary row */}
              <div
                className="disease-card-header"
                onClick={() => toggleExpand(disease.disease_id)}
                role="button"
                tabIndex={0}
              >
                <div className="disease-info-group">
                  <div className="disease-name-row">
                    <span className="disease-name">{disease.name}</span>
                    {disease.is_notifiable && (
                      <span className="notifiable-pill" title="Statutory notifiable disease — report to authorities">
                        <AlertTriangle size={12} /> Statutory Notifiable
                      </span>
                    )}
                  </div>
                  <span className="pathogen-label">Pathogen: {disease.pathogen}</span>
                </div>

                <div className="disease-score-group">
                  <div
                    className="likelihood-badge"
                    style={{
                      backgroundColor: badgeStyle.bg,
                      borderColor: badgeStyle.border,
                      color: badgeStyle.text,
                    }}
                  >
                    <span>{disease.likelihood} Match ({disease.probability_pct}%)</span>
                  </div>
                  <button className="expand-icon-btn" aria-label="Toggle disease details">
                    {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                  </button>
                </div>
              </div>

              {/* Collapsible Details */}
              {isExpanded && (
                <div className="disease-card-body">
                  {/* Acoustic Rationale */}
                  <div className="detail-section rationale-section">
                    <span className="section-label">Acoustic Signature Match:</span>
                    <p className="rationale-text">{disease.acoustic_rationale}</p>
                  </div>

                  {/* Physical Symptom Verification Checklist */}
                  <div className="detail-section">
                    <span className="section-label">Coop Inspection Checklist (Verify in Flock):</span>
                    <div className="symptom-checklist">
                      {disease.key_symptoms.map((symptom, sIdx) => {
                        const symptomKey = `${disease.disease_id}_${sIdx}`;
                        const isChecked = !!checkedSymptoms[symptomKey];

                        return (
                          <div
                            key={sIdx}
                            className={`checklist-item ${isChecked ? "checked" : ""}`}
                            onClick={() => toggleSymptom(symptomKey)}
                          >
                            {isChecked ? (
                              <CheckSquare size={16} className="check-icon text-primary" />
                            ) : (
                              <Square size={16} className="check-icon" />
                            )}
                            <span className="checklist-text">{symptom}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Biosecurity & Isolation Protocols */}
                  <div className="detail-section">
                    <span className="section-label">Actionable Farm Biosecurity Protocol:</span>
                    <ul className="biosecurity-list">
                      {disease.biosecurity_actions.map((action, aIdx) => (
                        <li key={aIdx} className="biosecurity-item">
                          <span className="bullet-point">▸</span>
                          <span>{action}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* General Biosecurity Advisory Footer */}
      {diseaseDifferential.overall_biosecurity_advice && (
        <div className="overall-biosecurity-card">
          <div className="biosecurity-header">
            <ShieldCheck size={18} className="text-emerald-400" />
            <h4 className="biosecurity-title">Immediate Flock Management Guidelines</h4>
          </div>
          <ul className="biosecurity-guidelines-list">
            {diseaseDifferential.overall_biosecurity_advice.map((item, idx) => (
              <li key={idx}>• {item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
