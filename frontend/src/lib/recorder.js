/**
 * Cross-browser audio recorder with MIME detection and live audio meter support.
 */

const CANDIDATE_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4", // Safari / iOS Safari
  "audio/ogg;codecs=opus",
  "audio/wav",
];

export function getSupportedMimeType() {
  if (typeof MediaRecorder === "undefined") return undefined;
  for (const type of CANDIDATE_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return undefined; // Let browser fall back to internal default
}

export class CoopRecorder {
  constructor({ maxDurationMs = 30000, onStop, onError, onLevelChange } = {}) {
    this.maxDurationMs = maxDurationMs;
    this.onStop = onStop;
    this.onError = onError;
    this.onLevelChange = onLevelChange;
    this.mediaRecorder = null;
    this.stream = null;
    this.audioContext = null;
    this.analyser = null;
    this.animationId = null;
    this.chunks = [];
    this.timer = null;
    this.startTime = 0;
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
        },
      });
    } catch (err) {
      this.onError?.(err);
      return;
    }

    // Initialize audio analyzer for live visualizer
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.audioContext = new AudioCtx();
        const source = this.audioContext.createMediaStreamSource(this.stream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 64;
        source.connect(this.analyser);
        this._pollAudioLevel();
      }
    } catch (e) {
      console.warn("Live visualizer context not available:", e);
    }

    const mimeType = getSupportedMimeType();
    const options = mimeType ? { mimeType } : undefined;

    try {
      this.mediaRecorder = new MediaRecorder(this.stream, options);
    } catch (e) {
      // Fallback without explicit options
      this.mediaRecorder = new MediaRecorder(this.stream);
    }

    this.chunks = [];
    this.startTime = Date.now();

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        this.chunks.push(e.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      this._cleanup();
      const actualType = this.mediaRecorder?.mimeType || mimeType || "audio/wav";
      const blob = new Blob(this.chunks, { type: actualType });
      const durationSec = Math.round((Date.now() - this.startTime) / 1000);
      this.onStop?.(blob, durationSec);
    };

    this.mediaRecorder.start(250); // Slice data every 250ms
    this.timer = setTimeout(() => this.stop(), this.maxDurationMs);
  }

  _pollAudioLevel() {
    if (!this.analyser) return;
    const data = new Uint8Array(this.analyser.frequencyBinCount);

    const check = () => {
      if (!this.analyser) return;
      this.analyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i];
      const avg = sum / data.length;
      const normalizedLevel = Math.min(1.0, avg / 128);
      this.onLevelChange?.(normalizedLevel, data);
      this.animationId = requestAnimationFrame(check);
    };
    check();
  }

  _cleanup() {
    clearTimeout(this.timer);
    if (this.animationId) cancelAnimationFrame(this.animationId);

    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close().catch(() => {});
    }
    this.audioContext = null;
    this.analyser = null;

    if (this.stream) {
      // Explicitly release mic tracks so the browser mic indicator turns off
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
  }

  stop() {
    clearTimeout(this.timer);
    if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
      this.mediaRecorder.stop();
    } else {
      this._cleanup();
    }
  }
}
