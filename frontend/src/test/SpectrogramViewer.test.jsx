import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import SpectrogramViewer from "../components/SpectrogramViewer";
import en from "../i18n/en";

describe("SpectrogramViewer Component", () => {
  const mockWindowData = {
    window_index: 0,
    start_sec: 0.0,
    end_sec: 5.0,
    risk_score: 75.0,
    ood_score: 1.25,
    is_ood: false,
    spectrogram_image: "data:image/jpeg;base64,/9j/mockBaseSpec",
    heatmap_image: "data:image/png;base64,iVBORmockBaseCam",
    biomarkers: {
      rale_intensity_pct: 62.0,
      spectral_centroid_hz: 2150.0,
      spectral_flatness: 0.012,
      event_density_pct: 55.0,
    },
  };

  it("renders spectrogram container, time axes, and frequency markers", () => {
    render(<SpectrogramViewer windowData={mockWindowData} t={en} />);

    expect(screen.getByText(/Acoustic Mel-Spectrogram & AI Attention/i)).toBeInTheDocument();
    expect(screen.getByText("8.0 kHz")).toBeInTheDocument();
    expect(screen.getByText(/4.5 kHz \[Wheeze\]/i)).toBeInTheDocument();
    expect(screen.getByText(/1.5 kHz \[Rale\]/i)).toBeInTheDocument();
    expect(screen.getByText("0.0s")).toBeInTheDocument();
    expect(screen.getByText("5.0s")).toBeInTheDocument();
  });

  it("toggles Grad-CAM heatmap overlay visibility on button click", () => {
    render(<SpectrogramViewer windowData={mockWindowData} t={en} />);

    const toggleBtn = screen.getByTitle("Toggle Grad-CAM AI Attention Overlay");
    expect(screen.getByAltText("Grad-CAM Attention Heatmap")).toBeInTheDocument();

    // Toggle off
    fireEvent.click(toggleBtn);
    expect(screen.queryByAltText("Grad-CAM Attention Heatmap")).not.toBeInTheDocument();

    // Toggle back on
    fireEvent.click(toggleBtn);
    expect(screen.getByAltText("Grad-CAM Attention Heatmap")).toBeInTheDocument();
  });

  it("adjusts heatmap opacity when range slider changes", () => {
    render(<SpectrogramViewer windowData={mockWindowData} t={en} />);

    const slider = screen.getByRole("slider");
    fireEvent.change(slider, { target: { value: "0.5" } });

    const heatmapImg = screen.getByAltText("Grad-CAM Attention Heatmap");
    expect(heatmapImg).toHaveStyle({ opacity: "0.5" });
  });
});
