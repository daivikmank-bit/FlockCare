import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ResultScreen from "../screens/ResultScreen";
import en from "../i18n/en";

describe("ResultScreen Topic-Based Multi-Page Component", () => {
  const onRecordAgainMock = vi.fn();

  const sampleResult = {
    risk_score: 82,
    risk_level: "high",
    message: "Elevated respiratory distress markers detected.",
    disclaimer: "This is a non-diagnostic screening tool.",
    windows_analyzed: 3,
    status: "calibrated",
    warning: null,
    ood_score: 1.15,
    windows_detail: [
      {
        window_index: 0,
        start_sec: 0.0,
        end_sec: 5.0,
        risk_score: 65.0,
        ood_score: 1.1,
        is_ood: false,
        spectrogram_image: "data:image/jpeg;base64,mock1",
        heatmap_image: "data:image/png;base64,mockHeat1",
        biomarkers: {
          rale_intensity_pct: 55.0,
          spectral_centroid_hz: 2100.0,
          spectral_flatness: 0.012,
          event_density_pct: 50.0,
        },
      },
    ],
    overall_biomarkers: {
      rale_intensity_pct: 61.0,
      spectral_centroid_hz: 2216.7,
      spectral_flatness: 0.0133,
      event_density_pct: 58.3,
    },
    feature_importance: [
      {
        feature_name: "High-Frequency Rale & Wheeze Power",
        value: "61.0%",
        impact: 35.0,
        direction: "increases_risk",
        clinical_significance: "Bronchial rale match.",
      },
    ],
    disease_differential: {
      flock_clinical_status: "Active Respiratory Distress",
      primary_concern: "Consistent with Infectious Bronchitis (IBV).",
      differentials: [
        {
          disease_id: "ibv",
          name: "Infectious Bronchitis (IBV)",
          pathogen: "Avian Coronavirus",
          likelihood: "High",
          probability_pct: 85,
          acoustic_rationale: "Tracheal wet rales match.",
          is_notifiable: false,
          key_symptoms: ["Watery eyes"],
          biosecurity_actions: ["Quarantine birds"],
        },
      ],
      overall_biosecurity_advice: ["Isolate birds"],
    },
  };

  it("renders Executive Overview page with risk score and Hers-style topic cards", () => {
    render(<ResultScreen result={sampleResult} onRecordAgain={onRecordAgainMock} t={en} />);

    expect(screen.getByText("Elevated Respiratory Distress Risk")).toBeInTheDocument();
    expect(screen.getByText("82%")).toBeInTheDocument();
    expect(screen.getByText("Detailed Topic Breakdowns")).toBeInTheDocument();
    expect(screen.getByText("Expected Avian Diseases")).toBeInTheDocument();
    expect(screen.getByText("Acoustic Saliency (Grad-CAM)")).toBeInTheDocument();
    expect(screen.getByText("Biomarkers & SHAP Factors")).toBeInTheDocument();
    expect(screen.getByText("Veterinary Care Plan")).toBeInTheDocument();
  });

  it("navigates to Expected Diseases topic page and back to Overview", () => {
    render(<ResultScreen result={sampleResult} onRecordAgain={onRecordAgainMock} t={en} />);

    const diseaseCard = screen.getByText("Expected Avian Diseases");
    fireEvent.click(diseaseCard);

    // Should navigate to topic page
    expect(screen.getAllByText("Expected Avian Disease Differential").length).toBeGreaterThan(0);
    expect(screen.getByText("Infectious Bronchitis (IBV)")).toBeInTheDocument();

    // Click back to Overview
    const backBtn = screen.getByRole("button", { name: "Back to Overview" });
    fireEvent.click(backBtn);
    expect(screen.getByText("Detailed Topic Breakdowns")).toBeInTheDocument();
  });

  it("navigates to Acoustic Saliency (Spectrogram & Grad-CAM) topic page", () => {
    render(<ResultScreen result={sampleResult} onRecordAgain={onRecordAgainMock} t={en} />);

    const saliencyCard = screen.getByText("Acoustic Saliency (Grad-CAM)");
    fireEvent.click(saliencyCard);

    expect(screen.getByText("Acoustic Saliency & Spectrogram")).toBeInTheDocument();
    expect(screen.getByText(/Acoustic Mel-Spectrogram & AI Attention/i)).toBeInTheDocument();
  });

  it("navigates to Biomarkers & SHAP topic page", () => {
    render(<ResultScreen result={sampleResult} onRecordAgain={onRecordAgainMock} t={en} />);

    const bmCard = screen.getByText("Biomarkers & SHAP Factors");
    fireEvent.click(bmCard);

    expect(screen.getByText("Acoustic Biomarkers & SHAP Factors")).toBeInTheDocument();
    expect(screen.getByText("Acoustic Biomarkers & SHAP Attribution")).toBeInTheDocument();
  });

  it("navigates to Veterinary Care Plan topic page", () => {
    render(<ResultScreen result={sampleResult} onRecordAgain={onRecordAgainMock} t={en} />);

    const vetCareCard = screen.getByText("Veterinary Care Plan");
    fireEvent.click(vetCareCard);

    expect(screen.getByText("Veterinary Care & Biosecurity Plan")).toBeInTheDocument();
    expect(screen.getByText("Clinical Veterinary PDF Report")).toBeInTheDocument();
  });

  it("opens Vet Report modal when 'Export Vet Report' button is clicked", () => {
    render(<ResultScreen result={sampleResult} onRecordAgain={onRecordAgainMock} t={en} />);

    const exportBtn = screen.getByText("Export Vet Report");
    fireEvent.click(exportBtn);

    expect(screen.getByText("Veterinary Clinical Acoustic Report")).toBeInTheDocument();
  });

  it("calls onRecordAgain when re-record button is clicked", () => {
    render(<ResultScreen result={sampleResult} onRecordAgain={onRecordAgainMock} t={en} />);

    const reRecordBtn = screen.getByText("Record Another Screening");
    fireEvent.click(reRecordBtn);
    expect(onRecordAgainMock).toHaveBeenCalledTimes(1);
  });
});
