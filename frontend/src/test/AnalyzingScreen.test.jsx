import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import AnalyzingScreen from "../screens/AnalyzingScreen";
import en from "../i18n/en";

describe("AnalyzingScreen Component", () => {
  it("renders radar container and analysis title", () => {
    render(<AnalyzingScreen t={en} />);
    expect(screen.getByText("Analyzing Coop Bioacoustics")).toBeInTheDocument();
    expect(screen.getByText(/Neural model scanning 5-second spectrogram windows/i)).toBeInTheDocument();
  });

  it("renders the 4 pipeline steps", () => {
    render(<AnalyzingScreen t={en} />);
    expect(screen.getByText(en.analyzingStep1)).toBeInTheDocument();
    expect(screen.getByText(en.analyzingStep2)).toBeInTheDocument();
    expect(screen.getByText(en.analyzingStep3)).toBeInTheDocument();
    expect(screen.getByText(en.analyzingStep4)).toBeInTheDocument();
  });
});
