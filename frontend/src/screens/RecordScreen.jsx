import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Upload, History, AlertTriangle, ShieldCheck, Activity, Info, CheckCircle2, Globe, User, LogOut } from "lucide-react";
import { CoopRecorder } from "../lib/recorder";

export default function RecordScreen({
  onComplete,
  error,
  t,
  currentLang,
  onToggleLang,
  history,
  onClearHistory,
  farmUser,
  onSignOut,
}) {
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
    <div className="card-shell record-screen-shell">
      {/* Header bar (Editorial Hers style with Logo) */}
      <header className="app-header">
        <div className="brand-group">
          <img
            src="/images/flockcare_logo.png"
            alt="FlockCare"
            className="brand-logo-xs"
          />
          <div className="lettermark-brand-sm">
            <span className="serif-brand">FlockCare</span>
          </div>
          {farmUser && (
            <div className="user-greeting-pill">
              <span className="greeting-label">{t.dashboardGreeting}</span>
              <span className="greeting-name">{farmUser}</span>
            </div>
          )}
        </div>

        <div className="header-actions">
          <button
            className="icon-pill-btn signout-pill-btn"
            onClick={onSignOut}
            title="Log out & Return to Starting Page"
          >
            <LogOut size={13} />
            <span>Log out</span>
          </button>

          <button
            className="icon-pill-btn"
            onClick={() => onToggleLang(currentLang === "en" ? "hi" : "en")}
            title="Toggle Language"
          >
            <Globe size={13} />
            <span>{currentLang === "en" ? "हिन्दी" : "EN"}</span>
          </button>

          <button
            className="icon-pill-btn"
            onClick={() => setShowHistory(!showHistory)}
            title={t.historyTitle}
          >
            <History size={14} />
            <span>{history?.length || 0}</span>
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

        {/* Guidance card with photography */}
        <div className="instruction-box">
          <div className="instruction-photo-row">
            <img
              src="/images/farmer_holding_hen.png"
              alt="Farmer inspecting poultry flock"
              className="instruction-thumb"
            />
            <div className="instruction-text-col">
              <div className="instruction-header">
                <h2 className="instruction-title">{t.recordHeading}</h2>
                <span className="instruction-pill">Acoustic AI</span>
              </div>
              <p className="instruction-body">{t.recordInstructions}</p>
            </div>
          </div>
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
                <Square size={28} className="stop-icon" />
                <span className="timer-count">{secondsElapsed}s</span>
              </div>
            ) : (
              <div className="fab-inner-idle">
                <Mic size={40} className="mic-icon" />
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
              <div className="progress-track">
                <div
                  className={`progress-fill ${isRecommendedDuration ? "fill-optimal" : "fill-recording"}`}
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <div className="progress-markers">
                <span className="marker marker-min">5s</span>
                <span className="marker marker-optimal">15s</span>
                <span className="marker marker-max">30s</span>
              </div>
              {isRecommendedDuration && (
                <div className="ready-indicator">
                  <CheckCircle2 size={14} />
                  <span>Optimal sampling duration reached</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Alternative file upload trigger */}
        {!recording && (
          <div className="upload-container">
            <input
              type="file"
              ref={fileInputRef}
              accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac,.aac"
              style={{ display: "none" }}
              onChange={handleFileSelect}
            />
            <button
              className="btn-upload"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={15} />
              <span>{t.uploadFile}</span>
            </button>
          </div>
        )}
      </main>

      {/* History Drawer Modal */}
      {showHistory && (
        <div className="modal-overlay" onClick={() => setShowHistory(false)}>
          <div className="modal-content history-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <History size={18} />
                <h3 className="modal-title">{t.historyTitle}</h3>
              </div>
              <button
                className="icon-btn-close"
                onClick={() => setShowHistory(false)}
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              {(!history || history.length === 0) ? (
                <div className="history-empty">
                  <p className="empty-title">{t.historyEmpty}</p>
                  <p className="empty-subtitle">{t.historyDeviceOnly}</p>
                </div>
              ) : (
                <div className="history-list">
                  {history.map((item, idx) => (
                    <div key={idx} className="history-card">
                      <div className="history-top">
                        <span className={`risk-pill pill-${item.risk_level}`}>
                          {item.risk_level.toUpperCase()} ({item.risk_score}%)
                        </span>
                        <span className="history-date">
                          {new Date(item.timestamp).toLocaleDateString()} {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                      <p className="history-msg">{item.message}</p>
                      <div className="history-meta">
                        <span>{item.windows_analyzed} Windows</span>
                        <span>•</span>
                        <span>Status: {item.status}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="modal-footer">
              <span className="history-device-note">{t.historyDeviceOnly}</span>
              {history && history.length > 0 && (
                <button className="danger-btn" onClick={onClearHistory}>
                  {t.clearHistory}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Footer info */}
      <footer className="app-footer">
        <ShieldCheck size={14} className="text-muted" />
        <p className="footer-disclaimer">{t.disclaimer}</p>
      </footer>
    </div>
  );
}
