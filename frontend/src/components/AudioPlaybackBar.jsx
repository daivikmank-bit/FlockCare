import React, { useState, useRef, useEffect } from "react";
import { Play, Pause, Volume2, RotateCcw } from "lucide-react";

export default function AudioPlaybackBar({ audioBlob, selectedWindowIndex, onSelectWindow, totalWindows, t }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const fallbackDuration = Math.max(15, (totalWindows || 3) * 5.0);
  const [duration, setDuration] = useState(fallbackDuration);
  const [audioUrl, setAudioUrl] = useState(null);

  const audioRef = useRef(null);

  useEffect(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [audioBlob]);

  function getSafeDuration() {
    if (
      audioRef.current &&
      isFinite(audioRef.current.duration) &&
      !isNaN(audioRef.current.duration) &&
      audioRef.current.duration > 0
    ) {
      return audioRef.current.duration;
    }
    return fallbackDuration;
  }

  function handlePlayPause() {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      const playPromise = audioRef.current.play();
      if (playPromise !== undefined && typeof playPromise.catch === "function") {
        playPromise.catch((err) => {
          console.warn("Audio playback error:", err);
        });
      }
      setIsPlaying(true);
    }
  }

  function handleTimeUpdate() {
    if (!audioRef.current) return;
    const curr = audioRef.current.currentTime;
    if (isFinite(curr) && !isNaN(curr) && curr >= 0) {
      setCurrentTime(curr);

      // Sync active window index
      const winCount = totalWindows || 3;
      const activeWindow = Math.min(winCount - 1, Math.max(0, Math.floor(curr / 5.0)));
      if (activeWindow !== selectedWindowIndex && onSelectWindow) {
        onSelectWindow(activeWindow);
      }
    }

    const safeDur = getSafeDuration();
    if (safeDur !== duration) {
      setDuration(safeDur);
    }
  }

  function handleSeek(e) {
    const seekTime = parseFloat(e.target.value);
    if (isFinite(seekTime) && !isNaN(seekTime)) {
      setCurrentTime(seekTime);
      if (audioRef.current) {
        try {
          audioRef.current.currentTime = seekTime;
        } catch (err) {
          console.warn("Seek error:", err);
        }
      }
    }
  }

  function handleLoadedMetadata() {
    const safeDur = getSafeDuration();
    setDuration(safeDur);
  }

  function formatTime(sec) {
    if (sec === null || sec === undefined || isNaN(sec) || !isFinite(sec) || sec < 0) {
      return "0:00";
    }
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  if (!audioUrl) return null;

  const displayDuration = getSafeDuration();

  return (
    <div className="audio-playback-bar">
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onDurationChange={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />

      <button
        className="play-pause-btn"
        onClick={handlePlayPause}
        aria-label={isPlaying ? "Pause audio" : "Play audio"}
      >
        {isPlaying ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
      </button>

      <div className="time-readout">
        <span>{formatTime(currentTime)}</span>
        <span>/</span>
        <span>{formatTime(displayDuration)}</span>
      </div>

      <input
        type="range"
        min="0"
        max={displayDuration}
        step="0.1"
        value={Math.min(currentTime, displayDuration)}
        onChange={handleSeek}
        className="audio-scrubber"
        aria-label="Audio scrubber"
      />

      <div className="audio-icon-group">
        <Volume2 size={15} className="text-muted" />
      </div>
    </div>
  );
}
