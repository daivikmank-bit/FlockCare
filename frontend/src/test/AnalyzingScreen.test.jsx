import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import AnalyzingScreen from "../screens/AnalyzingScreen";
import en from "../i18n/en";

describe("AnalyzingScreen Component", () => {
  it("renders radar container and analysis title", () => {
    render(<AnalyzingScreen t={en} />);
    expect(screen.getByText("Analyzing Coop Acoustics")).toBeInTheDocument();
    expect(screen.getByText(/Deep learning model is scanning 5-second spectrogram windows/i)).toBeInTheDocument();
  });

  it("renders all four multi-step analysis progress items", () => {
    render(<AnalyzingScreen t={en} />);
    expect(screen.getByText("Decoding audio container…")).toBeInTheDocument();
    expect(screen.getByText("Extracting log-mel spectrogram features…")).toBeInTheDocument();
    expect(screen.getByText("Evaluating acoustic domain & OOD gating…")).toBeInTheDocument();
    expect(screen.getByText("Synthesizing multi-window clinical risk…")).toBeInTheDocument();
  });
});
