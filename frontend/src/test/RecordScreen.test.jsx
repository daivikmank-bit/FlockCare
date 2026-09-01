import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import RecordScreen from "../screens/RecordScreen";
import en from "../i18n/en";

describe("RecordScreen Component", () => {
  const defaultProps = {
    onComplete: vi.fn(),
    error: null,
    t: en,
    currentLang: "en",
    onToggleLang: vi.fn(),
    history: [],
    onClearHistory: vi.fn(),
  };

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders idle state with branding, guidance, and instructions", () => {
    render(<RecordScreen {...defaultProps} />);
    expect(screen.getByText("FlockCare")).toBeInTheDocument();
    expect(screen.getByText("Coop Health Check")).toBeInTheDocument();
    expect(screen.getByText(/Hold your phone 1–2 meters from the flock/i)).toBeInTheDocument();
    expect(screen.getByText("Tap to Start Recording")).toBeInTheDocument();
  });

  it("renders error banner when an error message is passed", () => {
    render(
      <RecordScreen
        {...defaultProps}
        error="Microphone access was denied. Please allow microphone permissions."
      />
    );

    expect(screen.getByText(/Microphone access was denied/i)).toBeInTheDocument();
  });

  it("triggers file upload input handler on audio file selection", () => {
    const onCompleteMock = vi.fn();
    render(<RecordScreen {...defaultProps} onComplete={onCompleteMock} />);

    const fileInput = document.querySelector('input[type="file"]');
    const mockFile = new File(["audio dummy data"], "flock.wav", { type: "audio/wav" });
    fireEvent.change(fileInput, { target: { files: [mockFile] } });

    expect(onCompleteMock).toHaveBeenCalledWith(mockFile, "flock.wav");
  });

  it("opens history drawer when history button is clicked", () => {
    const mockHistory = [
      {
        timestamp: "2026-09-01T12:00:00Z",
        risk_score: 15.0,
        risk_level: "low",
        message: "Flock healthy",
        windows_analyzed: 3,
      },
    ];

    render(<RecordScreen {...defaultProps} history={mockHistory} />);

    const historyBtn = screen.getByTitle("Screening History");
    fireEvent.click(historyBtn);

    expect(screen.getByText("Screening History")).toBeInTheDocument();
    expect(screen.getByText(/Flock healthy/i)).toBeInTheDocument();
  });
});
