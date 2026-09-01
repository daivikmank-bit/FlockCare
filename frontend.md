# FlockCare — Part 6: Frontend (React)

*Standalone build guide. Assumes Part 5's API is deployed and reachable, with this exact contract:*
- `POST /analyze` — `multipart/form-data`, field name `file`, returns `{ risk_score, risk_level, message, disclaimer, windows_analyzed }`
- `GET /health` — `{ "status": "ok" }`

**Goal of this part:** Record → Analyzing → Result, working reliably across real phone browsers — not just desktop Chrome, where almost everything works by default and the actual gotchas don't show up.

---

## 6.0 Two things that will bite you if you skip them

1. **`MediaRecorder`'s output format depends on the browser, and Safari doesn't support webm.** Chrome/Firefox/Edge default to `audio/webm;codecs=opus`; Safari (and iOS Safari, which is what all iOS browsers actually run under the hood) produces `audio/mp4`. Hardcoding `audio/webm` will silently break for any iPhone user. Detect the supported type at runtime (6.3) — the good news is Part 5's backend already accepts both, so this is purely a frontend concern.
2. **`getUserMedia` requires a secure context.** It refuses to run over plain HTTP except on `localhost`. This won't show up in local dev (which is HTTP but exempt) — it'll show up the first time you share a preview link that isn't HTTPS, where the mic permission prompt just never appears and it looks like the app is broken. Make sure wherever you deploy the frontend serves HTTPS (Vercel/Netlify do this by default).

---

## 6.1 Setup
```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
```
`.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```

## 6.2 Structure
```
frontend/
├── src/
│   ├── screens/
│   │   ├── RecordScreen.jsx
│   │   ├── AnalyzingScreen.jsx
│   │   └── ResultScreen.jsx
│   ├── lib/
│   │   ├── recorder.js
│   │   ├── api.js
│   │   └── history.js
│   ├── i18n/
│   │   └── en.js
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
├── .env
└── package.json
```

---

## 6.3 Detecting a supported recording format
```javascript
// src/lib/recorder.js
const CANDIDATE_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",              // Safari / iOS
  "audio/ogg;codecs=opus",
];

export function getSupportedMimeType() {
  for (const type of CANDIDATE_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return undefined; // let the browser fall back to its own default
}
```

## 6.4 Recording
```javascript
// src/lib/recorder.js (continued)
export class CoopRecorder {
  constructor({ maxDurationMs = 30000, onStop, onError } = {}) {
    this.maxDurationMs = maxDurationMs;
    this.onStop = onStop;
    this.onError = onError;
    this.mediaRecorder = null;
    this.chunks = [];
    this.timer = null;
  }

  async start() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      // err.name: "NotAllowedError" (permission denied), "NotFoundError" (no mic),
      // "NotReadableError" (mic in use by another app) — handled in RecordScreen
      this.onError?.(err);
      return;
    }

    const mimeType = getSupportedMimeType();
    this.mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    this.chunks = [];

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
    this.mediaRecorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop()); // release the mic — see note below
      const blob = new Blob(this.chunks, { type: this.mediaRecorder.mimeType });
      this.onStop?.(blob);
    };

    this.mediaRecorder.start();
    this.timer = setTimeout(() => this.stop(), this.maxDurationMs);
  }

  stop() {
    clearTimeout(this.timer);
    if (this.mediaRecorder?.state === "recording") this.mediaRecorder.stop();
  }
}
```
Skipping `stream.getTracks().forEach(t => t.stop())` is an easy miss — without it, the browser's mic-in-use indicator stays on after recording ends, which reads as broken to a first-time user even though the app itself is working fine.

## 6.5 API client — matches Part 5's contract exactly
```javascript
// src/lib/api.js
const API_BASE = import.meta.env.VITE_API_BASE_URL;

export class AnalysisError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function analyzeRecording(blob) {
  const form = new FormData();
  const ext = blob.type.includes("mp4") ? "m4a" : "webm";
  form.append("file", blob, `coop_audio.${ext}`);   // field name MUST be "file" — Part 5 expects it

  const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new AnalysisError(body.detail || "Analysis failed.", res.status);
  }
  return res.json(); // { risk_score, risk_level, message, disclaimer, windows_analyzed }
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
```

---

## 6.6 App state machine
```jsx
// src/App.jsx
import { useState, useCallback } from "react";
import { analyzeRecording, AnalysisError } from "./lib/api";
import { saveToHistory } from "./lib/history";
import RecordScreen from "./screens/RecordScreen";
import AnalyzingScreen from "./screens/AnalyzingScreen";
import ResultScreen from "./screens/ResultScreen";
import "./App.css";

const SCREENS = { RECORD: "record", ANALYZING: "analyzing", RESULT: "result" };

export default function App() {
  const [screen, setScreen] = useState(SCREENS.RECORD);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleRecordingComplete = useCallback(async (blob) => {
    setScreen(SCREENS.ANALYZING);
    setError(null);
    try {
      const data = await analyzeRecording(blob);
      setResult(data);
      saveToHistory(data);
      setScreen(SCREENS.RESULT);
    } catch (err) {
      setError(err instanceof AnalysisError ? err.message : "Something went wrong. Try again.");
      setScreen(SCREENS.RECORD);
    }
  }, []);

  const reset = () => { setResult(null); setScreen(SCREENS.RECORD); };

  if (screen === SCREENS.ANALYZING) return <AnalyzingScreen />;
  if (screen === SCREENS.RESULT) return <ResultScreen result={result} onRecordAgain={reset} />;
  return <RecordScreen onComplete={handleRecordingComplete} error={error} />;
}
```

## 6.7 Record screen — with real error paths, not just the happy path
```jsx
// src/screens/RecordScreen.jsx
import { useState, useRef } from "react";
import { CoopRecorder } from "../lib/recorder";

const ERROR_MESSAGES = {
  NotAllowedError: "Microphone access was denied. Check your browser's site settings and try again.",
  NotFoundError: "No microphone was found on this device.",
  NotReadableError: "The microphone is already in use by another app.",
};

export default function RecordScreen({ onComplete, error }) {
  const [recording, setRecording] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(30);
  const [micError, setMicError] = useState(null);
  const recorderRef = useRef(null);

  async function handleStart() {
    setMicError(null);
    setSecondsLeft(30);
    const recorder = new CoopRecorder({
      maxDurationMs: 30000,
      onStop: onComplete,
      onError: (err) => setMicError(ERROR_MESSAGES[err.name] || "Couldn't access the microphone."),
    });
    recorderRef.current = recorder;
    await recorder.start();
    setRecording(true);
    const interval = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) { clearInterval(interval); setRecording(false); }
        return s - 1;
      });
    }, 1000);
  }

  function handleStop() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  return (
    <div className="screen">
      <h1>Coop health check</h1>
      <p>Hold your phone near the coop and record 10–30 seconds of ambient sound.</p>

      {(micError || error) && <p className="error-banner">{micError || error}</p>}

      <button
        className={`record-button ${recording ? "recording" : ""}`}
        onClick={recording ? handleStop : handleStart}
        aria-label={recording ? "Stop recording" : "Start recording"}
      >
        {recording ? secondsLeft : "🎙️"}
      </button>

      <p>{recording ? "Recording… tap to stop early" : "Tap to record"}</p>
    </div>
  );
}
```

## 6.8 Analyzing screen
```jsx
// src/screens/AnalyzingScreen.jsx
export default function AnalyzingScreen() {
  return (
    <div className="screen">
      <div className="spinner" />
      <p>Analyzing coop sounds…</p>
    </div>
  );
}
```

## 6.9 Result screen — color-coded by `risk_level`, matching your mockup
```jsx
// src/screens/ResultScreen.jsx
const RISK_STYLES = {
  low: { color: "#2e7d32", icon: "✅", label: "Flock is Healthy" },
  moderate: { color: "#f9a825", icon: "⚠️", label: "Some Signs of Stress" },
  high: { color: "#c62828", icon: "🚨", label: "Elevated Risk Detected" },
};

export default function ResultScreen({ result, onRecordAgain }) {
  const style = RISK_STYLES[result.risk_level] || RISK_STYLES.moderate;
  const vetSearchUrl = "https://www.google.com/maps/search/veterinarian+near+me";

  return (
    <div className="screen" style={{ borderTopColor: style.color }}>
      <span className="risk-icon">{style.icon}</span>
      <h2 style={{ color: style.color }}>{style.label}</h2>
      <p>{result.message}</p>
      <p className="disclaimer">{result.disclaimer}</p>

      {result.risk_level !== "low" && (
        <a href={vetSearchUrl} target="_blank" rel="noreferrer" className="vet-link">
          Find a nearby veterinarian
        </a>
      )}

      <button onClick={onRecordAgain}>Check again</button>
    </div>
  );
}
```
The "find a vet" link is a static Maps search, not the directory integration from your deck's mockup — that's a real feature (Part 12 in the master plan), not something to fake here. A working link to something useful beats a placeholder that goes nowhere.

## 6.10 Lightweight local history
No login in v1, so history is device-local rather than account-based:
```javascript
// src/lib/history.js
const KEY = "flockcare_history";
const MAX_ENTRIES = 20;

export function saveToHistory(result) {
  const existing = JSON.parse(localStorage.getItem(KEY) || "[]");
  const entry = { ...result, timestamp: Date.now() };
  localStorage.setItem(KEY, JSON.stringify([entry, ...existing].slice(0, MAX_ENTRIES)));
}

export function getHistory() {
  return JSON.parse(localStorage.getItem(KEY) || "[]");
}
```
Worth a one-line note in the UI ("history is saved on this device only") so a farmer doesn't assume it's backed up anywhere or visible from another phone.

## 6.11 Multilingual output
The backend returns English `message`/`disclaimer` text. Two options — pick one:
- **Frontend-owned (recommended for v1):** ignore the backend's English string, look up your own copy keyed by `risk_level`. No backend changes needed, and you already have `risk_level` for color-coding.
- **Backend-owned:** send a `lang` field, have the backend return pre-translated text. More consistent long-term, more work now.

```javascript
// src/i18n/en.js
export default {
  low: "Flock sounds healthy. No signs of respiratory distress detected.",
  moderate: "Some signs of respiratory stress. Monitor closely for 24–48 hours.",
  high: "Elevated respiratory sounds detected. Isolate the flock and consult a veterinarian.",
  disclaimer: "This is a screening tool, not a diagnosis.",
};
```
Add a second file (e.g. `hi.js`) with the same keys for your target regional language, and switch between them with a simple manual toggle — don't build a full i18n framework for a v1 demo.

## 6.12 Minimal styling
```css
/* src/App.css */
.screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  padding: 24px;
  text-align: center;
  border-top: 6px solid transparent;
}
.record-button {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  font-size: 2rem;
  background: #1976d2;
  color: white;
  border: none;
}
.record-button.recording { background: #c62828; }
.error-banner { color: #c62828; font-weight: 600; }
.disclaimer { font-size: 0.85rem; color: #666; margin-top: 12px; }
.vet-link { margin-top: 16px; display: inline-block; }
```
Large tap targets, color over dense text, three taps end to end — your farmer persona has variable literacy and likely a low-end Android phone, so lean on the icon/color pattern from your mockup rather than paragraphs of instructions.

---

## 6.13 Verification checklist
- [ ] Tested on an actual Android phone, not just desktop Chrome — desktop won't catch mobile permission-prompt quirks
- [ ] Tested on iOS Safari specifically if iPhone users are in scope — confirm the recorded `audio/mp4` blob actually plays back and analyzes correctly, don't just assume it matches Part 5's handling
- [ ] Deployed frontend is served over HTTPS — a non-HTTPS preview link will silently fail to prompt for mic access (6.0)
- [ ] Permission-denied and no-microphone paths show an actual message, not a stuck spinner
- [ ] The browser's "mic in use" indicator clears after recording stops — confirms track cleanup (6.4) is actually running
- [ ] A real round trip against the deployed Part 5 backend (not a mocked response) renders correctly for both a low-risk and a high-risk sample
- [ ] Recording a very quiet/silent clip surfaces Part 5's "recording too short" error cleanly in the UI, not a generic failure

---

## Where this leaves you
Parts 3–6 now form a complete, deployable v1: data pipeline → trained model → API → frontend. From the original 12-part plan, what's left is Part 8 (deployment — you've already got the Dockerfile from Part 5, this is wiring up hosting), Part 9 (testing & field validation), Part 10 (roadmap), Part 11 (risks), and Part 12 (future enhancements) — Part 7 (API contract) is already covered by Part 5's handoff section.

Say which one you want built out standalone next.
