import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import AudioPlaybackBar from "../components/AudioPlaybackBar";
import en from "../i18n/en";

describe("AudioPlaybackBar Component", () => {
  const dummyBlob = new Blob(["dummy audio bytes"], { type: "audio/webm" });

  it("renders audio player with safe finite duration and no NaN or Infinity", () => {
    render(
      <AudioPlaybackBar
        audioBlob={dummyBlob}
        selectedWindowIndex={0}
        onSelectWindow={() => {}}
        totalWindows={3}
        t={en}
      />
    );

    expect(screen.getByLabelText(/Play audio/i)).toBeInTheDocument();
    // Default format should be 0:00 / 0:15 without NaN or Infinity
    expect(screen.getByText("0:00")).toBeInTheDocument();
    expect(screen.getByText("0:15")).toBeInTheDocument();

    const scrubber = screen.getByLabelText("Audio scrubber");
    expect(scrubber).toBeInTheDocument();
    expect(scrubber).toHaveAttribute("max", "15");
  });

  it("handles play/pause toggle safely", () => {
    render(
      <AudioPlaybackBar
        audioBlob={dummyBlob}
        selectedWindowIndex={0}
        onSelectWindow={() => {}}
        totalWindows={3}
        t={en}
      />
    );

    const playBtn = screen.getByLabelText(/Play audio/i);
    fireEvent.click(playBtn);
    expect(playBtn).toBeInTheDocument();
  });
});
