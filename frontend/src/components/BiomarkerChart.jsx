import React from "react";
import { Activity, Zap, TrendingUp, TrendingDown, HelpCircle, BarChart3 } from "lucide-react";

export default function BiomarkerChart({ biomarkers, featureImportance, t }) {
  if (!biomarkers) return null;

  return (
    <div className="biomarker-section">
      <div className="biomarker-header">
        <Activity size={18} className="text-primary" />
        <h3 className="section-title">Acoustic Biomarkers & SHAP Attribution</h3>
      </div>

      {/* 4-Stat Metric Cards Grid */}
      <div className="biomarker-grid">
        {/* Rale & Wheeze Intensity */}
        <div className="biomarker-card">
          <div className="bm-card-header">
            <span className="bm-label">Tracheal Rale Power (1.5–4.5 kHz)</span>
            <span className="bm-badge">{biomarkers.rale_intensity_pct}%</span>
          </div>
          <div className="bm-progress-bar">
            <div
              className="bm-progress-fill"
              style={{
                width: `${Math.min(100, biomarkers.rale_intensity_pct)}%`,
                backgroundColor: biomarkers.rale_intensity_pct > 35 ? "#ef4444" : "#10b981",
              }}
            />
          </div>
          <span className="bm-subtext">Baseline: &lt;20% in healthy flocks</span>
        </div>

        {/* Spectral Centroid */}
        <div className="biomarker-card">
          <div className="bm-card-header">
            <span className="bm-label">Spectral Centroid (Sharpness)</span>
            <span className="bm-badge">{biomarkers.spectral_centroid_hz} Hz</span>
          </div>
          <div className="bm-progress-bar">
            <div
              className="bm-progress-fill"
              style={{
                width: `${Math.min(100, (biomarkers.spectral_centroid_hz / 3000) * 100)}%`,
                backgroundColor: biomarkers.spectral_centroid_hz > 1800 ? "#f59e0b" : "#10b981",
              }}
            />
          </div>
          <span className="bm-subtext">Normal range: 1200–1600 Hz</span>
        </div>

        {/* Event Density */}
        <div className="biomarker-card">
          <div className="bm-card-header">
            <span className="bm-label">Respiratory Event Density</span>
            <span className="bm-badge">{biomarkers.event_density_pct}%</span>
          </div>
          <div className="bm-progress-bar">
            <div
              className="bm-progress-fill"
              style={{
                width: `${Math.min(100, biomarkers.event_density_pct)}%`,
                backgroundColor: biomarkers.event_density_pct > 40 ? "#ef4444" : "#10b981",
              }}
            />
          </div>
          <span className="bm-subtext">Burst repetition rate across coop</span>
        </div>

        {/* Spectral Flatness */}
        <div className="biomarker-card">
          <div className="bm-card-header">
            <span className="bm-label">Harmonic Flatness</span>
            <span className="bm-badge">{biomarkers.spectral_flatness}</span>
          </div>
          <div className="bm-progress-bar">
            <div
              className="bm-progress-fill"
              style={{
                width: `${Math.min(100, biomarkers.spectral_flatness * 1000)}%`,
                backgroundColor: "#3b82f6",
              }}
            />
          </div>
          <span className="bm-subtext">Noise vs tonal harmonic balance</span>
        </div>
      </div>

      {/* SHAP Feature Importance Waterfall Bars */}
      {featureImportance && featureImportance.length > 0 && (
        <div className="shap-container">
          <div className="shap-header">
            <BarChart3 size={16} className="text-primary" />
            <h4 className="shap-title">AI Decision Factors (SHAP Attribution)</h4>
          </div>
          <p className="shap-description">
            Shows how specific acoustic characteristics pushed the prediction higher (red) or lower (green):
          </p>

          <div className="shap-bars-list">
            {featureImportance.map((feat, idx) => {
              const isPositive = feat.impact >= 0;
              const absVal = Math.min(50, Math.abs(feat.impact));

              return (
                <div key={idx} className="shap-item">
                  <div className="shap-label-row">
                    <div className="shap-name-group">
                      {isPositive ? (
                        <TrendingUp size={14} className="text-rose-400 mr-1" />
                      ) : (
                        <TrendingDown size={14} className="text-emerald-400 mr-1" />
                      )}
                      <span className="shap-feature-name">{feat.feature_name}</span>
                    </div>
                    <div className="shap-val-group">
                      <span className="shap-raw-val">{feat.value}</span>
                      <span className={`shap-impact-val ${isPositive ? "impact-pos" : "impact-neg"}`}>
                        {isPositive ? `+${feat.impact}%` : `${feat.impact}%`}
                      </span>
                    </div>
                  </div>

                  {/* Horizontal Bar */}
                  <div className="shap-bar-track">
                    <div
                      className={`shap-bar-fill ${isPositive ? "bar-pos" : "bar-neg"}`}
                      style={{ width: `${(absVal / 50) * 100}%` }}
                    />
                  </div>
                  <span className="shap-explanation">{feat.clinical_significance}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
