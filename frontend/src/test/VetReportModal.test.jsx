import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import VetReportModal from "../components/VetReportModal";
import en from "../i18n/en";

describe("VetReportModal Component", () => {
  const mockResult = {
    risk_score: 84.0,
    risk_level: "high",
    message: "Elevated respiratory sounds detected.",
    disclaimer: "Non-diagnostic screening.",
    windows_analyzed: 3,
    status: "calibrated",
    overall_biomarkers: {
      rale_intensity_pct: 62.0,
      spectral_centroid_hz: 2300.0,
      spectral_flatness: 0.014,
      event_density_pct: 58.0,
    },
    disease_differential: {
      differentials: [
        {
          name: "Infectious Bronchitis (IBV)",
          pathogen: "Avian Coronavirus",
          likelihood: "High",
          probability_pct: 88,
          acoustic_rationale: "Tracheal rales match wet exudate.",
          key_symptoms: ["Watery eyes", "Deformed eggs"],
        },
      ],
    },
  };

  it("renders clinical report sheet with biomarkers and differential diagnosis", () => {
    const onCloseMock = vi.fn();
    render(<VetReportModal result={mockResult} onClose={onCloseMock} t={en} />);

    expect(screen.getByText("Veterinary Clinical Acoustic Report")).toBeInTheDocument();
    expect(screen.getByText("FlockCare Avian Health Screening")).toBeInTheDocument();
    expect(screen.getByText("84%")).toBeInTheDocument();
    expect(screen.getByText("Acoustic Biomarker Matrix")).toBeInTheDocument();
    expect(screen.getByText(/Infectious Bronchitis \(IBV\)/i)).toBeInTheDocument();
  });

  it("triggers window.print when print button is clicked", () => {
    const printSpy = vi.spyOn(window, "print").mockImplementation(() => {});
    const onCloseMock = vi.fn();

    render(<VetReportModal result={mockResult} onClose={onCloseMock} t={en} />);
    const printBtn = screen.getByText(/Print \/ Save PDF/i);
    fireEvent.click(printBtn);

    expect(printSpy).toHaveBeenCalled();
    printSpy.mockRestore();
  });

  it("calls onClose when close icon button is clicked", () => {
    const onCloseMock = vi.fn();
    render(<VetReportModal result={mockResult} onClose={onCloseMock} t={en} />);

    const closeBtn = screen.getByRole("button", { name: "Close modal" });
    fireEvent.click(closeBtn);

    expect(onCloseMock).toHaveBeenCalled();
  });
});
