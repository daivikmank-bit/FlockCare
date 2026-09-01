import React, { act } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App";
import * as api from "../lib/api";

describe("App Screen State Transitions and UI", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.spyOn(api, "checkHealth").mockResolvedValue(true);
  });

  it("renders the initial RecordScreen with branding and guidance", () => {
    render(<App />);
    expect(screen.getByText("FlockCare")).toBeInTheDocument();
    expect(screen.getByText("Coop Health Check")).toBeInTheDocument();
    expect(screen.getByText(/Hold your phone 1–2 meters/i)).toBeInTheDocument();
  });

  it("toggles language between English and Hindi", async () => {
    render(<App />);
    const langBtn = screen.getByTitle("Toggle Language");

    // Click to switch to Hindi
    await act(async () => {
      fireEvent.click(langBtn);
    });
    expect(screen.getByText("दड़बे की स्वास्थ्य जांच")).toBeInTheDocument();

    // Click to switch back to English
    await act(async () => {
      fireEvent.click(langBtn);
    });
    expect(screen.getByText("Coop Health Check")).toBeInTheDocument();
  });

  it("transitions to ResultScreen upon successful analysis", async () => {
    vi.spyOn(api, "analyzeRecording").mockResolvedValue({
      risk_score: 85.0,
      risk_level: "high",
      message: "Elevated respiratory distress detected.",
      disclaimer: "This is a screening tool.",
      windows_analyzed: 3,
      status: "calibrated",
    });

    render(<App />);

    // Simulate file upload trigger
    const fileInput = document.querySelector('input[type="file"]');
    const mockFile = new File(["audio dummy data"], "coop.wav", { type: "audio/wav" });
    fireEvent.change(fileInput, { target: { files: [mockFile] } });

    // Should transition to analyzing
    expect(screen.getByText(/Analyzing Coop Acoustics/i)).toBeInTheDocument();

    // Then resolve to result screen
    await waitFor(() => {
      expect(screen.getByText("Elevated Respiratory Risk")).toBeInTheDocument();
      expect(screen.getByText("85%")).toBeInTheDocument();
      expect(screen.getByText("Find Nearby Poultry Veterinarians")).toBeInTheDocument();
    });

    // Check again resets back to record screen
    fireEvent.click(screen.getByText("Record Another Screening"));
    expect(screen.getByText("Coop Health Check")).toBeInTheDocument();
  });
});
