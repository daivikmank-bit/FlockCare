import { describe, it, expect, vi, beforeEach } from "vitest";
import { analyzeRecording, checkHealth, AnalysisError } from "../lib/api";

describe("API Client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("sends FormData with field name 'file' and parses success response", async () => {
    const mockResponseData = {
      risk_score: 15.0,
      risk_level: "low",
      message: "Flock sounds healthy",
      disclaimer: "Screening only",
      windows_analyzed: 3,
      status: "calibrated",
    };

    let capturedFormData = null;
    global.fetch = vi.fn().mockImplementation((url, options) => {
      capturedFormData = options.body;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponseData),
      });
    });

    const mockBlob = new Blob(["mock audio data"], { type: "audio/webm" });
    const result = await analyzeRecording(mockBlob, "recording.webm");

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/analyze"),
      expect.objectContaining({ method: "POST" })
    );

    expect(capturedFormData).toBeInstanceOf(FormData);
    expect(capturedFormData.has("file")).toBe(true);
    expect(result.risk_level).toBe("low");
    expect(result.windows_analyzed).toBe(3);
  });

  it("throws AnalysisError with server detail when status is 400", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: "Recording too short" }),
    });

    const mockBlob = new Blob(["short"], { type: "audio/wav" });
    await expect(analyzeRecording(mockBlob)).rejects.toThrowError("Recording too short");
  });

  it("checks /health and returns boolean status", async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
    expect(await checkHealth()).toBe(true);

    global.fetch = vi.fn().mockRejectedValue(new Error("Network error"));
    expect(await checkHealth()).toBe(false);
  });
});
