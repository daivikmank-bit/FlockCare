import React from "react";
import { CheckCircle2, AlertTriangle, AlertOctagon, RotateCcw, MapPin, ShieldAlert, Layers, ExternalLink } from "lucide-react";

export default function ResultScreen({ result, onRecordAgain, t }) {
  const isHealthy = result.risk_level === "low";
  const isModerate = result.risk_level === "moderate";
  const isHigh = result.risk_level === "high";
  const isOutOfRange = result.status === "out_of_range";

  // Visual styling token mapping
  const riskConfig = {
    low: {
      themeClass: "theme-healthy",
      icon: CheckCircle2,
      badgeText: t.healthyTitle,
      colorHex: "#10b981",
    },
    moderate: {
      themeClass: "theme-stress",
      icon: AlertTriangle,
      badgeText: t.stressTitle,
      colorHex: "#f59e0b",
    },
    high: {
      themeClass: "theme-elevated",
      icon: AlertOctagon,
      badgeText: t.elevatedTitle,
      colorHex: "#ef4444",
    },
  }[result.risk_level] || {
    themeClass: "theme-stress",
    icon: AlertTriangle,
    badgeText: t.stressTitle,
    colorHex: "#f59e0b",
  };

  const StatusIcon = riskConfig.icon;
  const vetSearchUrl = "https://www.google.com/maps/search/poultry+veterinarian+near+me";

  return (
    <div className={`card-shell result-shell ${riskConfig.themeClass}`}>
      {/* Risk Result Card Header */}
      <div className="result-header">
        <div className="status-avatar">
          <StatusIcon size={52} className="status-avatar-icon" />
        </div>
        <h1 className="status-title">{riskConfig.badgeText}</h1>
        <div className="risk-meter-container">
          <div className="risk-meter-bg">
            <div
              className="risk-meter-fill"
              style={{
                width: `${Math.max(8, result.risk_score)}%`,
                backgroundColor: riskConfig.colorHex,
              }}
            />
          </div>
          <div className="risk-score-readout">
            <span className="risk-score-label">{t.riskScore}</span>
            <span className="risk-score-value" style={{ color: riskConfig.colorHex }}>
              {result.risk_score}%
            </span>
          </div>
        </div>
      </div>

      {/* Out-of-Range Acoustic Gating Notice */}
      {isOutOfRange && (
        <div className="banner ood-banner">
          <div className="ood-header">
            <ShieldAlert size={18} className="ood-icon" />
            <strong className="ood-title">{t.outOfRangeTitle}</strong>
          </div>
          <p className="ood-body">{result.warning || t.outOfRangeWarning}</p>
        </div>
      )}

      {/* Main Clinical Recommendation */}
      <div className="recommendation-card">
        <h3 className="card-section-title">Clinical Screening Recommendation</h3>
        <p className="recommendation-text">{result.message}</p>

        {/* Windows and metadata pills */}
        <div className="metadata-pill-row">
          <div className="meta-pill">
            <Layers size={14} />
            <span>
              {result.windows_analyzed} {t.windowsAnalyzed}
            </span>
          </div>
          {result.ood_score && (
            <div className="meta-pill">
              <span>Domain Metric: {result.ood_score}</span>
            </div>
          )}
        </div>
      </div>

      {/* Vet Search Action for non-healthy status */}
      {!isHealthy && (
        <a
          href={vetSearchUrl}
          target="_blank"
          rel="noreferrer"
          className="btn-secondary vet-search-btn"
        >
          <MapPin size={18} className="text-danger" />
          <span>{t.findVet}</span>
          <ExternalLink size={14} className="text-muted" />
        </a>
      )}

      {/* Primary Action Button */}
      <button className="btn-primary record-again-btn" onClick={onRecordAgain}>
        <RotateCcw size={18} />
        <span>{t.checkAgain}</span>
      </button>

      {/* Legal & Medical Disclaimer */}
      <footer className="result-disclaimer">
        <p>{result.disclaimer || t.disclaimer}</p>
      </footer>
    </div>
  );
}
