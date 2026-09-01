import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ResultScreen from "../screens/ResultScreen";
import en from "../i18n/en";

describe("ResultScreen Component", () => {
  const onRecordAgainMock = vi.fn();

  it("renders Low Risk results correctly with green badge and no urgent vet CTA", () => {
    const lowResult = {
      risk_score: 12,
      risk_level: "low",
      message: "Acoustic patterns within normal parameters. Flock sounds healthy.",
      disclaimer: "This is a non-diagnostic screening tool.",
      windows_analyzed: 3,
      status: "calibrated",
      warning: null,
    };

    render(<ResultScreen result={lowResult} onRecordAgain={onRecordAgainMock} t={en} />);

    expect(screen.getByText("Flock is Healthy")).toBeInTheDocument();
    expect(screen.getByText("12%")).toBeInTheDocument();
    expect(screen.getByText(/Acoustic patterns within normal parameters/i)).toBeInTheDocument();
    expect(screen.getByText(/3 Acoustic Windows Analyzed/i)).toBeInTheDocument();
    expect(screen.queryByText(/Find Nearby Poultry Veterinarians/i)).not.toBeInTheDocument();
  });

  it("renders Moderate Risk results with amber badge and vet search link", () => {
    const modResult = {
      risk_score: 55,
      risk_level: "moderate",
      message: "Mild respiratory stress markers detected.",
      disclaimer: "This is a non-diagnostic screening tool.",
      windows_analyzed: 3,
      status: "calibrated",
      warning: null,
    };

    render(<ResultScreen result={modResult} onRecordAgain={onRecordAgainMock} t={en} />);

    expect(screen.getByText("Some Signs of Respiratory Stress")).toBeInTheDocument();
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.getByText("Find Nearby Poultry Veterinarians")).toBeInTheDocument();
  });

  it("renders High Risk results with red badge, alert styling, and vet search link", () => {
    const highResult = {
      risk_score: 88,
      risk_level: "high",
      message: "Elevated respiratory distress markers detected.",
      disclaimer: "This is a non-diagnostic screening tool.",
      windows_analyzed: 3,
      status: "calibrated",
      warning: null,
    };

    render(<ResultScreen result={highResult} onRecordAgain={onRecordAgainMock} t={en} />);

    expect(screen.getByText("Elevated Respiratory Risk")).toBeInTheDocument();
    expect(screen.getByText("88%")).toBeInTheDocument();
    const vetLink = screen.getByText("Find Nearby Poultry Veterinarians");
    expect(vetLink).toBeInTheDocument();
    expect(vetLink.closest("a")).toHaveAttribute("href", expect.stringContaining("google.com/maps"));
  });

  it("renders Out-of-Range acoustic warning notice when status is out_of_range", () => {
    const oodResult = {
      risk_score: 62,
      risk_level: "moderate",
      message: "Unusual acoustic environment.",
      disclaimer: "This is a non-diagnostic screening tool.",
      windows_analyzed: 3,
      status: "out_of_range",
      warning: "Audio characteristics deviate significantly from training baseline.",
    };

    render(<ResultScreen result={oodResult} onRecordAgain={onRecordAgainMock} t={en} />);

    expect(screen.getByText("Acoustic Range Notice")).toBeInTheDocument();
    expect(screen.getByText(/Audio characteristics deviate significantly from training baseline/i)).toBeInTheDocument();
  });

  it("calls onRecordAgain when 'Record Another Screening' button is clicked", () => {
    const lowResult = {
      risk_score: 10,
      risk_level: "low",
      message: "Flock healthy",
      disclaimer: "Non-diagnostic",
      windows_analyzed: 3,
      status: "calibrated",
    };

    render(<ResultScreen result={lowResult} onRecordAgain={onRecordAgainMock} t={en} />);
    fireEvent.click(screen.getByText("Record Another Screening"));
    expect(onRecordAgainMock).toHaveBeenCalledTimes(1);
  });
});
