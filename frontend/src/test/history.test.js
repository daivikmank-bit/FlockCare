import { describe, it, expect, beforeEach } from "vitest";
import { saveToHistory, getHistory, clearHistory } from "../lib/history";

describe("Local Screening History", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("saves screening results to localStorage and retrieves them", () => {
    const entry = {
      risk_score: 82.5,
      risk_level: "high",
      message: "Elevated respiratory sounds detected",
      windows_analyzed: 3,
    };

    saveToHistory(entry);
    const history = getHistory();

    expect(history.length).toBe(1);
    expect(history[0].risk_score).toBe(82.5);
    expect(history[0].risk_level).toBe("high");
    expect(history[0].timestamp).toBeDefined();
  });

  it("caps maximum history entries at 20", () => {
    for (let i = 0; i < 25; i++) {
      saveToHistory({ risk_score: i, risk_level: "low", message: `Test ${i}`, windows_analyzed: 3 });
    }

    const history = getHistory();
    expect(history.length).toBe(20);
    expect(history[0].risk_score).toBe(24); // Most recent first
  });

  it("clears history cleanly", () => {
    saveToHistory({ risk_score: 10, risk_level: "low", message: "Healthy", windows_analyzed: 3 });
    expect(getHistory().length).toBe(1);

    clearHistory();
    expect(getHistory().length).toBe(0);
  });
});
