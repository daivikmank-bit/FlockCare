import { describe, it, expect, vi, beforeEach } from "vitest";
import { getSupportedMimeType, CoopRecorder } from "../lib/recorder";

describe("Recorder MIME detection and lifecycle", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("detects webm;codecs=opus when supported (Chrome/Edge/Firefox)", () => {
    global.MediaRecorder = {
      isTypeSupported: vi.fn((type) => type === "audio/webm;codecs=opus"),
    };
    expect(getSupportedMimeType()).toBe("audio/webm;codecs=opus");
  });

  it("detects audio/mp4 on Safari / iOS Safari when webm is unsupported", () => {
    global.MediaRecorder = {
      isTypeSupported: vi.fn((type) => type === "audio/mp4"),
    };
    expect(getSupportedMimeType()).toBe("audio/mp4");
  });

  it("returns undefined fallback if no candidate types are supported", () => {
    global.MediaRecorder = {
      isTypeSupported: vi.fn(() => false),
    };
    expect(getSupportedMimeType()).toBeUndefined();
  });

  it("guarantees all microphone stream tracks are stopped when recording stops", async () => {
    const mockTrack = { stop: vi.fn() };
    const mockStream = {
      getTracks: () => [mockTrack, mockTrack],
    };

    global.navigator.mediaDevices = {
      getUserMedia: vi.fn().mockResolvedValue(mockStream),
    };

    class MockMediaRecorder {
      constructor() {
        this.state = "inactive";
        this.mimeType = "audio/webm";
      }
      start() {
        this.state = "recording";
      }
      stop() {
        this.state = "inactive";
        this.onstop?.();
      }
    }

    global.MediaRecorder = MockMediaRecorder;
    global.MediaRecorder.isTypeSupported = vi.fn(() => true);

    const onStop = vi.fn();
    const recorder = new CoopRecorder({ onStop });

    await recorder.start();
    expect(recorder.mediaRecorder.state).toBe("recording");

    recorder.stop();
    expect(mockTrack.stop).toHaveBeenCalled();
    expect(onStop).toHaveBeenCalled();
  });
});
