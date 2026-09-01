import React, { useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  RotateCcw,
  MapPin,
  ShieldAlert,
  Layers,
  ExternalLink,
  Flame,
  Activity,
  FileText,
  Volume2,
  Stethoscope,
  ChevronRight,
  ArrowLeft,
  ShieldCheck,
  BarChart3,
  Shield,
} from "lucide-react";
import SpectrogramViewer from "../components/SpectrogramViewer";
import DiseaseDifferentialCard from "../components/DiseaseDifferentialCard";
import BiomarkerChart from "../components/BiomarkerChart";
import AudioPlaybackBar from "../components/AudioPlaybackBar";
import VetReportModal from "../components/VetReportModal";

export default function ResultScreen({ result, audioBlob, onRecordAgain, t }) {
  // Navigation sub-pages: 'overview' | 'differentials' | 'spectrogram' | 'biomarkers' | 'vetcare'
  const [currentTopic, setCurrentTopic] = useState("overview");
  const [selectedWindowIdx, setSelectedWindowIdx] = useState(0);
  const [showVetReport, setShowVetReport] = useState(false);

  const isHealthy = result.risk_level === "low";
  const isModerate = result.risk_level === "moderate";
  const isHigh = result.risk_level === "high";
  const isOutOfRange = result.status === "out_of_range";

  const windows = result.windows_detail || [];
  const activeWindow = windows[selectedWindowIdx] || windows[0];

  const topDisease =
    result.disease_differential?.differentials?.[0] || null;

  // Visual styling token mapping
  const riskConfig = {
    low: {
      themeClass: "theme-healthy",
      icon: CheckCircle2,
      badgeText: t.healthyTitle || "Flock is Healthy",
      colorHex: "#166534",
    },
    moderate: {
      themeClass: "theme-stress",
      icon: AlertTriangle,
      badgeText: t.stressTitle || "Signs of Respiratory Stress Detected",
      colorHex: "#B45309",
    },
    high: {
      themeClass: "theme-elevated",
      icon: AlertOctagon,
      badgeText: t.elevatedTitle || "Elevated Respiratory Distress Risk",
      colorHex: "#B91C1C",
    },
  }[result.risk_level] || {
    themeClass: "theme-stress",
    icon: AlertTriangle,
    badgeText: t.stressTitle || "Signs of Respiratory Stress Detected",
    colorHex: "#B45309",
  };

  const StatusIcon = riskConfig.icon;
  const vetSearchUrl = "https://www.google.com/maps/search/poultry+veterinarian+near+me";

  return (
    <div className={`card-shell result-shell ${riskConfig.themeClass}`}>
      {/* =========================================================================
          VIEW 1: EXECUTIVE OVERVIEW PAGE (HERS-STYLE DASHBOARD WITH TOPIC CARDS)
          ========================================================================= */}
      {currentTopic === "overview" && (
        <div className="topic-page fade-in">
          {/* Top Bar with Report Export & Re-record */}
          <div className="result-top-bar">
            <button
              className="btn-pill-report"
              onClick={() => setShowVetReport(true)}
              title="Export formatted clinical report for poultry veterinarian"
            >
              <FileText size={14} />
              <span>Export Vet Report</span>
            </button>

            <button
              className="icon-pill-btn"
              onClick={onRecordAgain}
              title={t.checkAgain || "Record Another Screening"}
            >
              <RotateCcw size={14} />
            </button>
          </div>

          {/* Assessment Score Card */}
          <div className="result-header">
            <div className="status-avatar">
              <StatusIcon size={44} style={{ color: riskConfig.colorHex }} />
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
                <span className="risk-score-label">{t.riskScore || "Risk Level"}</span>
                <span className="risk-score-value" style={{ color: riskConfig.colorHex }}>
                  {result.risk_score}%
                </span>
              </div>
            </div>
          </div>

          {/* Out of calibrated range notice */}
          {isOutOfRange && (
            <div className="banner out-of-range-banner">
              <ShieldAlert size={18} className="banner-icon text-amber-500" />
              <div>
                <h4 className="banner-heading">{t.outOfRangeTitle || "Acoustic Range Notice"}</h4>
                <p className="banner-text">
                  {result.warning ||
                    t.outOfRangeWarning ||
                    "Audio characteristics deviate from calibrated baseline. Recommendation: Re-record closer to the birds."}
                </p>
              </div>
            </div>
          )}

          {/* Audio Playback Bar */}
          {audioBlob && (
            <AudioPlaybackBar
              audioBlob={audioBlob}
              selectedWindowIndex={selectedWindowIdx}
              onSelectWindow={setSelectedWindowIdx}
              totalWindows={result.windows_analyzed || 3}
              t={t}
            />
          )}

          {/* Clinical Finding Summary */}
          <div className="recommendation-card">
            <h3 className="card-section-title">Clinical Assessment</h3>
            <p className="recommendation-text">{result.message}</p>
            <div className="metadata-pill-row">
              <div className="meta-pill">
                <Layers size={13} />
                <span>{result.windows_analyzed} {t.windowsAnalyzed || "Acoustic Windows"}</span>
              </div>
              {result.ood_score !== null && (
                <div className="meta-pill">
                  <span>OOD Metric: {result.ood_score}</span>
                </div>
              )}
            </div>
          </div>

          {/* Topic Navigation Section (Hers Reference Multi-Page Layout) */}
          <div className="result-topic-section">
            <h3 className="topic-section-heading">Detailed Topic Breakdowns</h3>
            <div className="topic-card-grid">
              {/* Card 1: Disease Differentials */}
              <div
                className="topic-nav-card"
                onClick={() => setCurrentTopic("differentials")}
                role="button"
                tabIndex={0}
              >
                <div className="topic-card-icon-circle">
                  <Stethoscope size={18} />
                </div>
                <div className="topic-card-text">
                  <div className="topic-card-title-row">
                    <span className="topic-card-title">Expected Avian Diseases</span>
                    {topDisease && (
                      <span className="topic-badge">
                        {topDisease.probability_pct}% Match
                      </span>
                    )}
                  </div>
                  <p className="topic-card-desc">
                    {topDisease
                      ? `Predominant match: ${topDisease.name}`
                      : "Differential diagnosis for IBV, CRD, Coryza & NDV"}
                  </p>
                </div>
                <ChevronRight size={16} className="topic-arrow" />
              </div>

              {/* Card 2: Acoustic AI & Saliency (Grad-CAM) */}
              <div
                className="topic-nav-card"
                onClick={() => setCurrentTopic("spectrogram")}
                role="button"
                tabIndex={0}
              >
                <div className="topic-card-icon-circle">
                  <Flame size={18} />
                </div>
                <div className="topic-card-text">
                  <div className="topic-card-title-row">
                    <span className="topic-card-title">Acoustic Saliency (Grad-CAM)</span>
                  </div>
                  <p className="topic-card-desc">
                    Mel-spectrogram heatmap showing tracheal rale & wheeze attention
                  </p>
                </div>
                <ChevronRight size={16} className="topic-arrow" />
              </div>

              {/* Card 3: Biomarkers & SHAP Decision Factors */}
              <div
                className="topic-nav-card"
                onClick={() => setCurrentTopic("biomarkers")}
                role="button"
                tabIndex={0}
              >
                <div className="topic-card-icon-circle">
                  <Activity size={18} />
                </div>
                <div className="topic-card-text">
                  <div className="topic-card-title-row">
                    <span className="topic-card-title">Biomarkers & SHAP Factors</span>
                    {result.overall_biomarkers && (
                      <span className="topic-badge">
                        {result.overall_biomarkers.rale_intensity_pct}% Rale
                      </span>
                    )}
                  </div>
                  <p className="topic-card-desc">
                    Tracheal power, spectral centroid, and positive/negative AI attributions
                  </p>
                </div>
                <ChevronRight size={16} className="topic-arrow" />
              </div>

              {/* Card 4: Veterinary Action Plan */}
              <div
                className="topic-nav-card"
                onClick={() => setCurrentTopic("vetcare")}
                role="button"
                tabIndex={0}
              >
                <div className="topic-card-icon-circle">
                  <ShieldCheck size={18} />
                </div>
                <div className="topic-card-text">
                  <div className="topic-card-title-row">
                    <span className="topic-card-title">Veterinary Care Plan</span>
                  </div>
                  <p className="topic-card-desc">
                    Coop biosecurity protocols, vet finder, and printable clinical report
                  </p>
                </div>
                <ChevronRight size={16} className="topic-arrow" />
              </div>
            </div>
          </div>

          {/* Record Another Screening CTA */}
          <button className="btn-pill-solid" onClick={onRecordAgain} style={{ marginTop: 14 }}>
            <RotateCcw size={16} />
            <span>{t.checkAgain || "Record Another Screening"}</span>
          </button>
        </div>
      )}

      {/* =========================================================================
          VIEW 2: DEDICATED TOPIC PAGES (DISEASES, SPECTROGRAM, BIOMARKERS, VET CARE)
          ========================================================================= */}
      {currentTopic !== "overview" && (
        <div className="topic-page fade-in">
          {/* Sub-page Navigation Header */}
          <header className="topic-subpage-header">
            <button
              className="topic-back-btn"
              onClick={() => setCurrentTopic("overview")}
              aria-label="Back to Overview"
            >
              <ArrowLeft size={16} />
              <span>Overview</span>
            </button>

            <div className="topic-pill-tabs">
              <button
                className={`topic-tab-pill ${currentTopic === "differentials" ? "active" : ""}`}
                onClick={() => setCurrentTopic("differentials")}
              >
                Diseases
              </button>
              <button
                className={`topic-tab-pill ${currentTopic === "spectrogram" ? "active" : ""}`}
                onClick={() => setCurrentTopic("spectrogram")}
              >
                Saliency
              </button>
              <button
                className={`topic-tab-pill ${currentTopic === "biomarkers" ? "active" : ""}`}
                onClick={() => setCurrentTopic("biomarkers")}
              >
                Biomarkers
              </button>
              <button
                className={`topic-tab-pill ${currentTopic === "vetcare" ? "active" : ""}`}
                onClick={() => setCurrentTopic("vetcare")}
              >
                Vet Care
              </button>
            </div>
          </header>

          {/* 1. TOPIC: EXPECTED AVIAN DISEASES */}
          {currentTopic === "differentials" && (
            <div className="topic-content-body">
              <div className="topic-intro-banner">
                <h2 className="topic-page-title">Expected Avian Disease Differential</h2>
                <p className="topic-page-desc">
                  Bioacoustic match analysis comparing flock rale patterns against common avian respiratory pathogens.
                </p>
              </div>

              <DiseaseDifferentialCard
                diseaseDifferential={result.disease_differential}
                t={t}
              />
            </div>
          )}

          {/* 2. TOPIC: SPECTROGRAM & GRAD-CAM SALIENCY */}
          {currentTopic === "spectrogram" && (
            <div className="topic-content-body">
              <div className="topic-intro-banner">
                <h2 className="topic-page-title">Acoustic Saliency & Spectrogram</h2>
                <p className="topic-page-desc">
                  Neural model convolutional feature attention (Grad-CAM) mapped over 5-second log-mel spectrogram windows.
                </p>
              </div>

              {/* Multi-Window Timeline Selector */}
              {windows.length > 0 && (
                <div className="window-timeline-selector">
                  <span className="timeline-title">Select 5-Second Window:</span>
                  <div className="window-pill-row">
                    {windows.map((w, idx) => {
                      const isSelected = idx === selectedWindowIdx;
                      const wRiskColor =
                        w.risk_score > 60 ? "#B91C1C" : w.risk_score > 35 ? "#B45309" : "#166534";

                      return (
                        <button
                          key={idx}
                          className={`window-pill ${isSelected ? "active" : ""}`}
                          onClick={() => setSelectedWindowIdx(idx)}
                        >
                          <div className="pill-top">
                            <span className="pill-name">Window {idx + 1}</span>
                            <span className="pill-time">{w.start_sec}s–{w.end_sec}s</span>
                          </div>
                          <div className="pill-bottom">
                            <span className="pill-score" style={{ color: wRiskColor }}>
                              {w.risk_score}% Risk
                            </span>
                            {w.is_ood && <span className="pill-ood-tag">OOD</span>}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {activeWindow && (
                <>
                  <SpectrogramViewer windowData={activeWindow} t={t} />

                  {activeWindow.biomarkers && (
                    <div className="window-biomarker-summary">
                      <div className="w-bm-item">
                        <span className="w-bm-lbl">Window Rale Power:</span>
                        <span className="w-bm-val">{activeWindow.biomarkers.rale_intensity_pct}%</span>
                      </div>
                      <div className="w-bm-item">
                        <span className="w-bm-lbl">Spectral Centroid:</span>
                        <span className="w-bm-val">{activeWindow.biomarkers.spectral_centroid_hz} Hz</span>
                      </div>
                      <div className="w-bm-item">
                        <span className="w-bm-lbl">Event Density:</span>
                        <span className="w-bm-val">{activeWindow.biomarkers.event_density_pct}%</span>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* 3. TOPIC: BIOMARKERS & SHAP DECISION FACTORS */}
          {currentTopic === "biomarkers" && (
            <div className="topic-content-body">
              <div className="topic-intro-banner">
                <h2 className="topic-page-title">Acoustic Biomarkers & SHAP Factors</h2>
                <p className="topic-page-desc">
                  Quantitative bioacoustic metrics and directional SHAP feature importance explaining the neural prediction.
                </p>
              </div>

              <BiomarkerChart
                biomarkers={result.overall_biomarkers}
                featureImportance={result.feature_importance}
                t={t}
              />
            </div>
          )}

          {/* 4. TOPIC: VETERINARY CARE & BIOSECURITY PLAN */}
          {currentTopic === "vetcare" && (
            <div className="topic-content-body">
              <div className="topic-intro-banner">
                <h2 className="topic-page-title">Veterinary Care & Biosecurity Plan</h2>
                <p className="topic-page-desc">
                  Recommended flock containment guidelines and direct access to poultry veterinary resources.
                </p>
              </div>

              {/* Local Vet Search Card */}
              <div className="vet-action-card">
                <div className="vet-card-header">
                  <MapPin size={22} className="text-amber-600" />
                  <div>
                    <h4 className="vet-heading">{t.findVet || "Find Nearby Poultry Veterinarians"}</h4>
                    <p className="vet-subtext">
                      Locate licensed avian veterinarians and livestock extension diagnostic clinics.
                    </p>
                  </div>
                </div>
                <a
                  href={vetSearchUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-vet-search"
                >
                  <span>{t.findVet || "Find Nearby Poultry Veterinarians"}</span>
                  <ExternalLink size={15} />
                </a>
              </div>

              {/* Printable PDF Report Launcher Card */}
              <div className="printable-report-cta-card">
                <div className="cta-card-header">
                  <FileText size={22} className="text-dark" />
                  <div>
                    <h4 className="cta-heading">Clinical Veterinary PDF Report</h4>
                    <p className="cta-subtext">
                      Generate a formatted document with biomarker tables and spectrogram snapshots for your vet.
                    </p>
                  </div>
                </div>
                <button
                  className="btn-pill-solid"
                  onClick={() => setShowVetReport(true)}
                  style={{ marginTop: 8 }}
                >
                  <FileText size={16} />
                  <span>Generate & Print Clinical Report</span>
                </button>
              </div>
            </div>
          )}

          {/* Back to Overview button at bottom of each topic */}
          <button
            className="btn-pill-outline"
            onClick={() => setCurrentTopic("overview")}
            style={{ marginTop: 16 }}
          >
            <ArrowLeft size={16} />
            <span>Return to Assessment Overview</span>
          </button>
        </div>
      )}

      {/* Footer Disclaimer */}
      <footer className="result-disclaimer">
        <p>{result.disclaimer || t.disclaimer}</p>
      </footer>

      {/* Vet Report Export Modal */}
      {showVetReport && (
        <VetReportModal
          result={result}
          onClose={() => setShowVetReport(false)}
          t={t}
        />
      )}
    </div>
  );
}
