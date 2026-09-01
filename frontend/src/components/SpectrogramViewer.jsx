import React, { useState } from "react";
import { Eye, Layers, Flame, Info } from "lucide-react";

export default function SpectrogramViewer({ windowData, t }) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.75);

  if (!windowData) return null;

  return (
    <div className="spectrogram-container">
      {/* Header Controls */}
      <div className="spectrogram-header">
        <div className="spectrogram-title-group">
          <Layers size={16} className="text-primary" />
          <span className="spectrogram-title">
            Acoustic Mel-Spectrogram & AI Attention (Window {windowData.window_index + 1}: {windowData.start_sec}s–{windowData.end_sec}s)
          </span>
        </div>

        <div className="spectrogram-controls">
          <button
            className={`spec-toggle-btn ${showHeatmap ? "active" : ""}`}
            onClick={() => setShowHeatmap(!showHeatmap)}
            title="Toggle Grad-CAM AI Attention Overlay"
          >
            <Flame size={14} />
            <span>Grad-CAM Saliency</span>
          </button>

          {showHeatmap && (
            <div className="opacity-slider-group">
              <span className="slider-label">Opacity: {Math.round(heatmapOpacity * 100)}%</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                className="opacity-slider"
              />
            </div>
          )}
        </div>
      </div>

      {/* Visual Canvas Wrapper */}
      <div className="spec-canvas-wrapper">
        {/* Frequency Y-Axis Labels */}
        <div className="freq-axis">
          <span>8.0 kHz</span>
          <span className="rale-band-label">4.5 kHz [Wheeze]</span>
          <span className="rale-band-label">1.5 kHz [Rale]</span>
          <span>0 Hz</span>
        </div>

        {/* Image Layer Stack */}
        <div className="spec-image-stack">
          {/* Base Spectrogram Image */}
          <img
            src={windowData.spectrogram_image}
            alt="Mel-Spectrogram"
            className="spec-layer spec-base"
          />

          {/* Grad-CAM Saliency Overlay */}
          {showHeatmap && (
            <img
              src={windowData.heatmap_image}
              alt="Grad-CAM Attention Heatmap"
              className="spec-layer spec-heatmap"
              style={{ opacity: heatmapOpacity }}
            />
          )}

          {/* Rale Target Band Guide Lines */}
          <div className="rale-guide-zone" />
        </div>
      </div>

      {/* Time X-Axis */}
      <div className="time-axis">
        <span>{windowData.start_sec}.0s</span>
        <span>{windowData.start_sec + 1.25}s</span>
        <span>{windowData.start_sec + 2.5}s</span>
        <span>{windowData.start_sec + 3.75}s</span>
        <span>{windowData.end_sec}.0s</span>
      </div>

      {/* Legend and Interpretation Footer */}
      <div className="spec-legend-bar">
        <div className="legend-item">
          <span className="legend-swatch spec-swatch" />
          <span>Mel Acoustic Energy (Magma)</span>
        </div>
        <div className="legend-item">
          <span className="legend-swatch cam-swatch" />
          <span>AI Attention Hotspot (Grad-CAM Saliency)</span>
        </div>
        <div className="legend-item rale-legend">
          <span className="legend-indicator" />
          <span>Tracheal Rale Target Zone (1.5–4.5 kHz)</span>
        </div>
      </div>
    </div>
  );
}
