/**
 * Local device history storage for screening records.
 */

const KEY = "flockcare_history";
const MAX_ENTRIES = 20;

export function saveToHistory(result) {
  try {
    const existing = JSON.parse(localStorage.getItem(KEY) || "[]");
    const entry = {
      ...result,
      id: `flock_${Date.now()}`,
      timestamp: Date.now(),
    };
    const updated = [entry, ...existing].slice(0, MAX_ENTRIES);
    localStorage.setItem(KEY, JSON.stringify(updated));
    return entry;
  } catch (e) {
    console.warn("Could not save to localStorage:", e);
    return null;
  }
}

export function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

export function clearHistory() {
  try {
    localStorage.removeItem(KEY);
  } catch (e) {
    console.warn("Could not clear localStorage:", e);
  }
}
