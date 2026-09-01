import React from "react";
import { X, Printer, Download, CheckCircle, AlertTriangle, ShieldCheck, Stethoscope } from "lucide-react";

export default function VetReportModal({ result, onClose, t }) {
  if (!result) return null;

  const now = new Date().toLocaleString();

  function handlePrint() {
    window.print();
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content report-modal" onClick={(e) => e.stopPropagation()}>
        {/* Modal Top Bar */}
        <div className="report-modal-header no-print">
          <div className="report-header-title">
            <Stethoscope size={20} className="text-primary" />
            <span>Veterinary Clinical Acoustic Report</span>
          </div>
          <div className="report-header-actions">
            <button className="btn-secondary print-btn" onClick={handlePrint}>
              <Printer size={16} /> Print / Save PDF
            </button>
            <button className="icon-btn close-modal-btn" onClick={onClose} aria-label="Close modal">
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Printable Report Document Body */}
        <div className="printable-report-sheet">
          {/* Header */}
          <div className="sheet-header">
            <div>
              <h1 className="sheet-brand">FlockCare Avian Health Screening</h1>
              <p className="sheet-tagline">Coop Bioacoustic Diagnostics & Respiratory Differential Report</p>
            </div>
            <div className="sheet-meta">
              <div><strong>Date:</strong> {now}</div>
              <div><strong>Duration:</strong> {result.windows_analyzed * 5}s ({result.windows_analyzed} windows)</div>
              <div><strong>Acoustic Status:</strong> {result.status.toUpperCase()}</div>
            </div>
          </div>

          <hr className="sheet-divider" />

          {/* Risk Level Highlight */}
          <div className={`sheet-risk-banner risk-${result.risk_level}`}>
            <div className="banner-score">
              <span className="score-num">{result.risk_score}%</span>
              <span className="score-tier">Overall Respiratory Stress Index ({result.risk_level.toUpperCase()})</span>
            </div>
            <div className="banner-msg">{result.message}</div>
          </div>

          {/* Biomarkers Table */}
          {result.overall_biomarkers && (
            <div className="sheet-section">
              <h3 className="sheet-section-title">Acoustic Biomarker Matrix</h3>
              <table className="sheet-table">
                <thead>
                  <tr>
                    <th>Biomarker</th>
                    <th>Measured Value</th>
                    <th>Reference Range</th>
                    <th>Clinical Correlation</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Tracheal Rale Power (1.5–4.5 kHz)</td>
                    <td><strong>{result.overall_biomarkers.rale_intensity_pct}%</strong></td>
                    <td>&lt; 20%</td>
                    <td>Tracheal exudate & bronchial wet rattle</td>
                  </tr>
                  <tr>
                    <td>Spectral Centroid Shift</td>
                    <td><strong>{result.overall_biomarkers.spectral_centroid_hz} Hz</strong></td>
                    <td>1200–1600 Hz</td>
                    <td>Inspiratory wheezing frequency sharpness</td>
                  </tr>
                  <tr>
                    <td>Respiratory Event Density</td>
                    <td><strong>{result.overall_biomarkers.event_density_pct}%</strong></td>
                    <td>&lt; 25%</td>
                    <td>Flock coughing & snicking recurrence</td>
                  </tr>
                  <tr>
                    <td>Harmonic Flatness</td>
                    <td><strong>{result.overall_biomarkers.spectral_flatness}</strong></td>
                    <td>&lt; 0.01</td>
                    <td>Loss of brooding tonality / noise distortion</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {/* Expected Disease Differentials */}
          {result.disease_differential && result.disease_differential.differentials && (
            <div className="sheet-section">
              <h3 className="sheet-section-title">Differential Diagnosis & Symptom Checklist</h3>
              <div className="sheet-diff-list">
                {result.disease_differential.differentials.slice(0, 3).map((d, i) => (
                  <div key={i} className="sheet-diff-card">
                    <div className="diff-card-top">
                      <span className="diff-card-name">{i + 1}. {d.name} ({d.probability_pct}% Match - {d.likelihood})</span>
                      <span className="diff-card-pathogen">{d.pathogen}</span>
                    </div>
                    <p className="diff-card-rationale"><strong>Acoustic Match:</strong> {d.acoustic_rationale}</p>
                    <div className="diff-card-symptoms">
                      <strong>Check for in flock:</strong> {d.key_symptoms.join(" • ")}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Biosecurity Instructions */}
          <div className="sheet-section">
            <h3 className="sheet-section-title">Recommended Clinical & Biosecurity Protocols</h3>
            <ul className="sheet-bullet-list">
              <li>Isolate symptomatic birds in a warm, dry quarantine coop immediately.</li>
              <li>Perform tracheal and choanal swabbing for PCR/culture confirmation before broad antibiotic use.</li>
              <li>Sanitize communal drinkers with approved virucidal disinfectants (e.g. Virkon S).</li>
              <li>Maintain strict farm biosecurity: sanitize boots and equipment between pens.</li>
            </ul>
          </div>

          {/* Disclaimer */}
          <footer className="sheet-footer">
            <p><strong>Disclaimer:</strong> FlockCare is an AI screening tool designed to augment poultry management. This report does not replace formal clinical diagnosis by a licensed veterinarian.</p>
          </footer>
        </div>
      </div>
    </div>
  );
}
