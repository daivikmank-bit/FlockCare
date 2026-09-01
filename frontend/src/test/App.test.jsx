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

  it("renders the initial LandingScreen with classic lettermark and hero typography", () => {
    render(<App />);
    expect(screen.getByText("flockcare")).toBeInTheDocument();
    expect(screen.getByText(/Better care designed just for/i)).toBeInTheDocument();
    expect(screen.getByText("your flock")).toBeInTheDocument();
    expect(screen.getByText("Get started")).toBeInTheDocument();
    expect(screen.getByText("Log in")).toBeInTheDocument();
  });

  it("navigates to SignInScreen on 'Log in' and back on back arrow", () => {
    render(<App />);
    const logInBtn = screen.getByText("Log in");
    fireEvent.click(logInBtn);

    expect(screen.getByText("Welcome back")).toBeInTheDocument();
    expect(screen.getByText("Continue as Guest Farmer")).toBeInTheDocument();

    // Click back button
    const backBtn = screen.getByRole("button", { name: /Back to starting page/i });
    fireEvent.click(backBtn);
    expect(screen.getByText("Get started")).toBeInTheDocument();
  });

  it("transitions through SignIn to RecordScreen and performs full audio screening", async () => {
    vi.spyOn(api, "analyzeRecording").mockResolvedValue({
      risk_score: 85.0,
      risk_level: "high",
      message: "Elevated respiratory distress detected.",
      disclaimer: "This is a screening tool.",
      windows_analyzed: 3,
      status: "calibrated",
      windows_detail: [],
    });

    render(<App />);

    // Click Get started -> leads to SignIn
    const getStartedBtn = screen.getByText("Get started");
    fireEvent.click(getStartedBtn);

    expect(screen.getByText("Welcome back")).toBeInTheDocument();

    // Bypass with guest farmer or fill form
    const guestBtn = screen.getByText("Continue as Guest Farmer");
    fireEvent.click(guestBtn);

    expect(screen.getByText("Coop Health Screening")).toBeInTheDocument();

    // Simulate file upload trigger
    const fileInput = document.querySelector('input[type="file"]');
    const mockFile = new File(["audio dummy data"], "coop.wav", { type: "audio/wav" });
    fireEvent.change(fileInput, { target: { files: [mockFile] } });

    // Should transition to analyzing
    expect(screen.getByText(/Analyzing Coop Bioacoustics/i)).toBeInTheDocument();

    // Then resolve to result screen overview with topic cards
    await waitFor(() => {
      expect(screen.getByText("Elevated Respiratory Distress Risk")).toBeInTheDocument();
      expect(screen.getByText("Expected Avian Diseases")).toBeInTheDocument();
      expect(screen.getByText("Veterinary Care Plan")).toBeInTheDocument();
    });

    // Check again resets back to record screen
    fireEvent.click(screen.getByText("Record Another Screening"));
    expect(screen.getByText("Coop Health Screening")).toBeInTheDocument();
  });
});
