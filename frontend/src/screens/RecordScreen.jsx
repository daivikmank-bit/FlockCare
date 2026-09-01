import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Upload, History, AlertTriangle, ShieldCheck, Activity, Info, CheckCircle2 } from "lucide-react";
import { CoopRecorder } from "../lib/recorder";

export default function RecordScreen({ onComplete, error, t, currentLang, onToggleLang, history, onClearHistory }) {
  const [recording, setRecording] = useState(false);
  const [secondsElapsed, setSecondsElapsed] = useState(0);
  const [micError, setMicError] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [showHistory, setShowHistory] = useState(false);

  const recorderRef = useRef(null);
  const timerRef = useRef(null);
  const fileInputRef = useRef(null);

  const MAX_SECONDS = 30;
  const MIN_SECONDS = 5;

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      recorderRef.current?.stop();
    };
  }, []);

  async function handleStartRecording() {
    setMicError(null);
    setSecondsElapsed(0);

    const recorder = new CoopRecorder({
      maxDurationMs: MAX_SECONDS * 1000,
      onStop: (blob, durationSec) => {
        setRecording(false);
        if (timerRef.current) clearInterval(timerRef.current);
        onComplete(blob);
      },
      onError: (err) => {
        setRecording(false);
        if (timerRef.current) clearInterval(timerRef.current);
        if (err.name === "NotAllowedError") {
          setMicError(t.micErrorDenied);
        } else if (err.name === "NotFoundError") {
          setMicError(t.micErrorNotFound);
        } else if (err.name === "NotReadableError") {
          setMicError(t.micErrorBusy);
        } else {
          setMicError(err.message || t.micErrorDenied);
        }
      },
      onLevelChange: (level) => {
        setAudioLevel(level);
      },
    });

    recorderRef.current = recorder;
    await recorder.start();
    setRecording(true);

    timerRef.current = setInterval(() => {
      setSecondsElapsed((prev) => {
        if (prev >= MAX_SECONDS - 1) {
          handleStopRecording();
          return MAX_SECONDS;
        }
        return prev + 1;
      });
    }, 1000);
  }

  function handleStopRecording() {
    if (timerRef.current) clearInterval(timerRef.current);
    if (recorderRef.current) {
      recorderRef.current.stop();
    }
    setRecording(false);
  }

  function handleFileSelect(e) {
    const file = e.target.files?.[0];
    if (file) {
      onComplete(file, file.name);
    }
  }

  const progressPct = (secondsElapsed / MAX_SECONDS) * 100;
  const isRecommendedDuration = secondsElapsed >= 15;
  const isMinDurationMet = secondsElapsed >= MIN_SECONDS;

  return (
    <div className="card-shell">
      {/* Header bar */}
      <header className="app-header">
        <div className="brand-group">
          <div className="brand-icon">🐔</div>
          <div>
            <div className="brand-title-row">
              <h1 className="brand-name">{t.appTitle}</h1>
              <span className="brand-badge">{t.appBadge}</span>
            </div>
            <p className="brand-subtitle">{t.subtitle}</p>
          </div>
        </div>

        <div className="header-actions">
          <button
            className="icon-btn"
            onClick={() => onToggleLang(currentLang === "en" ? "hi" : "en")}
            title="Toggle Language"
          >
            {currentLang === "en" ? "🇮🇳 हिन्दी" : "🇬🇧 English"}
          </button>
          <button
            className="icon-btn"
            onClick={() => setShowHistory(!showHistory)}
            title={t.historyTitle}
          >
            <History size={18} />
          </button>
        </div>
      </header>

      {/* Main interaction surface */}
      <main className="record-main">
        {/* Error notification banner */}
        {(micError || error) && (
          <div className="banner error-banner">
            <AlertTriangle size={18} className="banner-icon" />
            <div className="banner-text">{micError || error}</div>
          </div>
        )}

        {/* Guidance card */}
        <div className="instruction-box">
          <div className="instruction-header">
            <Activity size={18} className="text-primary" />
            <h2 className="instruction-title">{t.recordHeading}</h2>
          </div>
          <p className="instruction-body">{t.recordInstructions}</p>
          <div className="duration-tip">
            <Info size={14} />
            <span>{t.minDurationNote}</span>
          </div>
        </div>

        {/* Central Recording Target */}
        <div className="record-center">
          <div
            className={`record-halo ${recording ? "halo-active" : ""}`}
            style={{
              transform: recording ? `scale(${1.0 + audioLevel * 0.4})` : "scale(1)",
              opacity: recording ? 0.3 + audioLevel * 0.7 : 0.1,
            }}
          />

          <button
            className={`record-fab ${recording ? "recording-state" : "idle-state"}`}
            onClick={recording ? handleStopRecording : handleStartRecording}
            aria-label={recording ? t.tapToStop : t.tapToRecord}
          >
            {recording ? (
              <div className="fab-inner-recording">
                <Square size={32} className="stop-icon" />
                <span className="timer-count">{secondsElapsed}s</span>
              </div>
            ) : (
              <div className="fab-inner-idle">
                <Mic size={44} className="mic-icon" />
              </div>
            )}
          </button>
        </div>

        {/* Status prompt */}
        <div className="record-status-container">
          <p className="record-status-prompt">
            {recording ? t.recordingActive : t.tapToRecord}
          </p>

          {recording && (
            <div className="progress-container">
              <div className="progress-bar-bg">
                <div
                  className={`progress-bar-fill ${isRecommendedDuration ? "fill-good" : ""}`}
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <div className="progress-labels">
                <span className={isMinDurationMet ? "text-success font-medium" : "text-muted"}>
                  {isMinDurationMet ? "✓ 5s Min Met" : "5s Min"}
                </span>
                <span className={isRecommendedDuration ? "text-success font-medium" : "text-muted"}>
                  {isRecommendedDuration ? "✓ Optimal (15–30s)" : "30s Max"}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Alternative file upload */}
        {!recording && (
          <div className="upload-alternative">
            <input
              type="file"
              ref={fileInputRef}
              accept="audio/*,.wav,.webm,.ogg,.mp4,.mp3,.m4a"
              style={{ display: "none" }}
              onChange={handleFileSelect}
            />
            <button
              className="text-link-btn"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={16} />
              <span>{t.uploadFile}</span>
            </button>
          </div>
        )}
      </main>

      {/* History Drawer Modal */}
      {showHistory && (
        <div className="modal-backdrop" onClick={() => setShowHistory(false)}>
          <div className="modal-sheet" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <History size={20} />
                <h3>{t.historyTitle}</h3>
              </div>
              <button className="close-btn" onClick={() => setShowHistory(false)}>✕</button>
            </div>

            <div className="modal-body">
              {history.length === 0 ? (
                <div className="history-empty-state">
                  <p>{t.historyEmpty}</p>
                </div>
              ) : (
                <div className="history-list">
                  {history.map((item, idx) => (
                    <div key={item.id || idx} className={`history-card risk-${item.risk_level}`}>
                      <div className="history-top">
                        <span className={`risk-pill pill-${item.risk_level}`}>
                          {item.risk_level.toUpperCase()}
                        </span>
                        <span className="history-date">
                          {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      </div>
                      <p className="history-msg">{item.message}</p>
                      <div className="history-meta">
                        <span>Risk: {item.risk_score}%</span>
                        <span>•</span>
                        <span>{item.windows_analyzed} windows</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {history.length > 0 && (
              <div className="modal-footer">
                <span className="history-device-note">{t.historyDeviceOnly}</span>
                <button className="danger-btn" onClick={onClearHistory}>{t.clearHistory}</button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer disclaimer */}
      <footer className="app-footer">
        <ShieldCheck size={14} className="text-muted" />
        <p className="footer-disclaimer">{t.disclaimer}</p>
      </footer>
    </div>
  );
}
