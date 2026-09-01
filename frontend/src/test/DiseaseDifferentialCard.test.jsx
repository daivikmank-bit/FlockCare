import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import DiseaseDifferentialCard from "../components/DiseaseDifferentialCard";
import en from "../i18n/en";

describe("DiseaseDifferentialCard Component", () => {
  const mockDifferential = {
    flock_clinical_status: "Active Respiratory Distress Detected",
    primary_concern: "Acoustic profile indicates elevated risk predominantly consistent with Infectious Bronchitis (IBV).",
    differentials: [
      {
        disease_id: "ibv",
        name: "Infectious Bronchitis (IBV)",
        pathogen: "Avian Coronavirus",
        likelihood: "High",
        probability_pct: 88,
        acoustic_rationale: "High rale/wheeze energy (62.5%) strongly corresponds to wet tracheal exudate.",
        is_notifiable: false,
        key_symptoms: ["Watery eyes & nasal bubbles", "Wrinkled or thin-shelled eggs", "Coughing & snicking"],
        biosecurity_actions: ["Isolate sneezing birds into quarantine shed", "Disinfect waterers twice daily"],
      },
      {
        disease_id: "ndv",
        name: "Newcastle Disease (Respiratory Form)",
        pathogen: "Avian Paramyxovirus 1 (APMV-1)",
        likelihood: "Moderate",
        probability_pct: 54,
        acoustic_rationale: "Vocal fragmentation and high-distress gasping chirps.",
        is_notifiable: true,
        key_symptoms: ["Severe open-beak gasping", "Green watery diarrhea", "Twisted neck (torticollis)"],
        biosecurity_actions: ["Report immediately to animal health officer", "Implement complete farm lockdown"],
      },
    ],
    overall_biosecurity_advice: [
      "Quarantine birds exhibiting active audible wheezing.",
      "Sanitize communal water lines.",
    ],
  };

  it("renders differential title and disease names with matching percentages", () => {
    render(<DiseaseDifferentialCard diseaseDifferential={mockDifferential} t={en} />);

    expect(screen.getByText("Expected Avian Disease Differential")).toBeInTheDocument();
    expect(screen.getByText("Infectious Bronchitis (IBV)")).toBeInTheDocument();
    expect(screen.getByText("High Match (88%)")).toBeInTheDocument();
    expect(screen.getByText("Newcastle Disease (Respiratory Form)")).toBeInTheDocument();
    expect(screen.getByText("Moderate Match (54%)")).toBeInTheDocument();
  });

  it("renders statutory notifiable badge for notifiable diseases like Newcastle", () => {
    render(<DiseaseDifferentialCard diseaseDifferential={mockDifferential} t={en} />);
    expect(screen.getByText(/Statutory Notifiable/i)).toBeInTheDocument();
  });

  it("expands disease card on click to reveal symptoms checklist and biosecurity protocols", () => {
    render(<DiseaseDifferentialCard diseaseDifferential={mockDifferential} t={en} />);

    const ibvCardHeader = screen.getByText("Infectious Bronchitis (IBV)");
    fireEvent.click(ibvCardHeader);

    // Should reveal symptoms
    expect(screen.getByText(/Watery eyes & nasal bubbles/i)).toBeInTheDocument();
    expect(screen.getByText(/Isolate sneezing birds into quarantine shed/i)).toBeInTheDocument();

    // Toggle symptom checkbox
    const symptomItem = screen.getByText(/Watery eyes & nasal bubbles/i).closest(".checklist-item");
    fireEvent.click(symptomItem);
    expect(symptomItem).toHaveClass("checked");
  });
});
