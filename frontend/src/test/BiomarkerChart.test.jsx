import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import BiomarkerChart from "../components/BiomarkerChart";
import en from "../i18n/en";

describe("BiomarkerChart Component", () => {
  const mockBiomarkers = {
    rale_intensity_pct: 58.4,
    spectral_centroid_hz: 2240.0,
    spectral_flatness: 0.0152,
    event_density_pct: 64.0,
  };

  const mockFeatureImportance = [
    {
      feature_name: "High-Frequency Rale & Wheeze Power (1.5–4.5 kHz)",
      value: "58.4%",
      impact: 32.5,
      direction: "increases_risk",
      clinical_significance: "Key acoustic marker for wet bronchial secretions.",
    },
    {
      feature_name: "Baseline Flock Roosting Harmonics (<1 kHz)",
      value: "Disrupted",
      impact: 8.0,
      direction: "increases_risk",
      clinical_significance: "Healthy brooding vocalizations produce stable harmonics.",
    },
  ];

  it("renders 4 biomarker metric cards with measured values", () => {
    render(
      <BiomarkerChart
        biomarkers={mockBiomarkers}
        featureImportance={mockFeatureImportance}
        t={en}
      />
    );

    expect(screen.getByText("Acoustic Biomarkers & SHAP Attribution")).toBeInTheDocument();
    expect(screen.getAllByText("58.4%").length).toBeGreaterThan(0);
    expect(screen.getByText("2240 Hz")).toBeInTheDocument();
    expect(screen.getByText("64%")).toBeInTheDocument();
  });

  it("renders SHAP feature attribution items with positive/negative impacts", () => {
    render(
      <BiomarkerChart
        biomarkers={mockBiomarkers}
        featureImportance={mockFeatureImportance}
        t={en}
      />
    );

    expect(screen.getByText(/AI Decision Factors \(SHAP Attribution\)/i)).toBeInTheDocument();
    expect(screen.getByText("High-Frequency Rale & Wheeze Power (1.5–4.5 kHz)")).toBeInTheDocument();
    expect(screen.getByText("+32.5%")).toBeInTheDocument();
    expect(screen.getByText(/Key acoustic marker for wet bronchial secretions/i)).toBeInTheDocument();
  });
});
