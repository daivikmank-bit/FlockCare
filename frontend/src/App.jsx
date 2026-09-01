import React, { useState, useCallback, useEffect } from "react";
import { AlertCircle } from "lucide-react";
import { analyzeRecording, checkHealth, AnalysisError } from "./lib/api";
import { saveToHistory, getHistory, clearHistory } from "./lib/history";
import LandingScreen from "./screens/LandingScreen";
import SignInScreen from "./screens/SignInScreen";
import RecordScreen from "./screens/RecordScreen";
import AnalyzingScreen from "./screens/AnalyzingScreen";
import ResultScreen from "./screens/ResultScreen";
import enTranslations from "./i18n/en";
import hiTranslations from "./i18n/hi";
import "./App.css";

const SCREENS = {
  LANDING: "landing",
  SIGN_IN: "signin",
  RECORD: "record",
  ANALYZING: "analyzing",
  RESULT: "result",
};

export default function App() {
  const [screen, setScreen] = useState(() => {
    // If farm account remembered in localStorage, go to record directly, else show landing page
    const savedFarm = localStorage.getItem("flockcare_farm_user");
    return savedFarm ? SCREENS.RECORD : SCREENS.LANDING;
  });

  const [farmUser, setFarmUser] = useState(() => localStorage.getItem("flockcare_farm_user") || null);
  const [result, setResult] = useState(null);
  const [currentAudioBlob, setCurrentAudioBlob] = useState(null);
  const [error, setError] = useState(null);
  const [lang, setLang] = useState(() => localStorage.getItem("flockcare_lang") || "en");
  const [history, setHistory] = useState(() => getHistory());
  const [isBackendHealthy, setIsBackendHealthy] = useState(true);

  const t = lang === "hi" ? hiTranslations : enTranslations;

  useEffect(() => {
    // Check API health on startup, retrying every 3s if free server is waking up
    let intervalId;
    const verifyHealth = async () => {
      const ok = await checkHealth();
      setIsBackendHealthy(ok);
      if (ok && intervalId) {
        clearInterval(intervalId);
      }
    };

    verifyHealth();
    intervalId = setInterval(verifyHealth, 3500);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, []);

  const handleToggleLang = (newLang) => {
    setLang(newLang);
    localStorage.setItem("flockcare_lang", newLang);
  };

  const handleSignInSuccess = (userOrFarmName, remember = true) => {
    setFarmUser(userOrFarmName);
    if (remember) {
      localStorage.setItem("flockcare_farm_user", userOrFarmName);
    } else {
      localStorage.removeItem("flockcare_farm_user");
    }
    setScreen(SCREENS.RECORD);
  };

  const handleGuestContinue = () => {
    setFarmUser("Guest Farmer");
    localStorage.removeItem("flockcare_farm_user");
    setScreen(SCREENS.RECORD);
  };

  const handleSignOut = () => {
    localStorage.removeItem("flockcare_farm_user");
    setFarmUser(null);
    setScreen(SCREENS.LANDING);
  };

  const handleRecordingComplete = useCallback(async (blobOrFile, originalFilename) => {
    setScreen(SCREENS.ANALYZING);
    setError(null);
    setCurrentAudioBlob(blobOrFile);

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
    setCurrentAudioBlob(null);
    setError(null);
    setScreen(SCREENS.RECORD);
  };

  return (
    <div className="app-viewport">
      {!isBackendHealthy && (
        <div className="offline-banner">
          <AlertCircle size={14} />
          <span>{t.offlineWarning}</span>
        </div>
      )}

      {/* Screen 1: Starting / Landing Page (Hers inspired) */}
      {screen === SCREENS.LANDING && (
        <LandingScreen
          onGetStarted={() => setScreen(SCREENS.SIGN_IN)}
          onLogIn={() => setScreen(SCREENS.SIGN_IN)}
          t={t}
          currentLang={lang}
          onToggleLang={handleToggleLang}
        />
      )}

      {/* Screen 2: Sign-In Page */}
      {screen === SCREENS.SIGN_IN && (
        <SignInScreen
          onSignInSuccess={handleSignInSuccess}
          onBackToLanding={() => setScreen(SCREENS.LANDING)}
          onGuestContinue={handleGuestContinue}
          t={t}
        />
      )}

      {/* Screen 3: Analyzing State */}
      {screen === SCREENS.ANALYZING && <AnalyzingScreen t={t} />}

      {/* Screen 4: Comprehensive XAI Diagnostic Results */}
      {screen === SCREENS.RESULT && result && (
        <ResultScreen
          result={result}
          audioBlob={currentAudioBlob}
          onRecordAgain={resetToRecord}
          t={t}
        />
      )}

      {/* Screen 5: Coop Audio Recording Dashboard */}
      {screen === SCREENS.RECORD && (
        <RecordScreen
          onComplete={handleRecordingComplete}
          error={error}
          t={t}
          currentLang={lang}
          onToggleLang={handleToggleLang}
          history={history}
          onClearHistory={handleClearHistory}
          farmUser={farmUser}
          onSignOut={handleSignOut}
        />
      )}
    </div>
  );
}
