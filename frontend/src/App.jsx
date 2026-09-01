import React, { useState, useCallback, useEffect } from "react";
import { analyzeRecording, checkHealth, AnalysisError } from "./lib/api";
import { saveToHistory, getHistory, clearHistory } from "./lib/history";
import RecordScreen from "./screens/RecordScreen";
import AnalyzingScreen from "./screens/AnalyzingScreen";
import ResultScreen from "./screens/ResultScreen";
import enTranslations from "./i18n/en";
import hiTranslations from "./i18n/hi";
import "./App.css";

const SCREENS = {
  RECORD: "record",
  ANALYZING: "analyzing",
  RESULT: "result",
};

export default function App() {
  const [screen, setScreen] = useState(SCREENS.RECORD);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [lang, setLang] = useState(() => localStorage.getItem("flockcare_lang") || "en");
  const [history, setHistory] = useState(() => getHistory());
  const [isBackendHealthy, setIsBackendHealthy] = useState(true);

  const t = lang === "hi" ? hiTranslations : enTranslations;

  useEffect(() => {
    // Check API health on startup
    checkHealth().then((ok) => setIsBackendHealthy(ok));
  }, []);

  const handleToggleLang = (newLang) => {
    setLang(newLang);
    localStorage.setItem("flockcare_lang", newLang);
  };

  const handleRecordingComplete = useCallback(async (blobOrFile, originalFilename) => {
    setScreen(SCREENS.ANALYZING);
    setError(null);

    try {
      const data = await analyzeRecording(blobOrFile, originalFilename);
      setResult(data);
      saveToHistory(data);
      setHistory(getHistory());
      setScreen(SCREENS.RESULT);
    } catch (err) {
      console.error("Recording analysis error:", err);
      const msg = err instanceof AnalysisError ? err.message : "Analysis failed. Please try recording again.";
      setError(msg);
      setScreen(SCREENS.RECORD);
    }
  }, []);

  const handleClearHistory = () => {
    clearHistory();
    setHistory([]);
  };

  const resetToRecord = () => {
    setResult(null);
    setError(null);
    setScreen(SCREENS.RECORD);
  };

  return (
    <div className="app-viewport">
      {!isBackendHealthy && (
        <div className="offline-banner">
          <span>⚠️ {t.offlineWarning}</span>
        </div>
      )}

      {screen === SCREENS.ANALYZING && <AnalyzingScreen t={t} />}

      {screen === SCREENS.RESULT && result && (
        <ResultScreen result={result} onRecordAgain={resetToRecord} t={t} />
      )}

      {screen === SCREENS.RECORD && (
        <RecordScreen
          onComplete={handleRecordingComplete}
          error={error}
          t={t}
          currentLang={lang}
          onToggleLang={handleToggleLang}
          history={history}
          onClearHistory={handleClearHistory}
        />
      )}
    </div>
  );
}
