/**
 * API client for FlockCare backend inference and health checking.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class AnalysisError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
    this.name = "AnalysisError";
  }
}

export async function analyzeRecording(blob, originalFilename) {
  const form = new FormData();

  let filename = originalFilename;
  if (!filename) {
    let ext = "wav";
    if (blob.type.includes("webm")) ext = "webm";
    else if (blob.type.includes("mp4") || blob.type.includes("m4a")) ext = "m4a";
    else if (blob.type.includes("ogg")) ext = "ogg";
    filename = `coop_audio_${Date.now()}.${ext}`;
  }

  // Field name MUST be "file" to match backend expectation
  form.append("file", blob, filename);

  let res;
  try {
    res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      body: form,
    });
  } catch (netErr) {
    throw new AnalysisError(
      "Cannot connect to the analysis server. Please check your internet connection or verify the backend is running.",
      0
    );
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const message = body.detail || `Server error (${res.status})`;
    throw new AnalysisError(message, res.status);
  }

  return await res.json();
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}
