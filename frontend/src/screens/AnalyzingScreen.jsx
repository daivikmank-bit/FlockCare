import React, { useState, useEffect } from "react";
import { Loader2, Waves, Cpu, Sparkles, CheckCircle } from "lucide-react";

export default function AnalyzingScreen({ t }) {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { label: t.analyzingStep1, icon: Waves },
    { label: t.analyzingStep2, icon: Cpu },
    { label: t.analyzingStep3, icon: Sparkles },
    { label: t.analyzingStep4, icon: CheckCircle },
  ];

  useEffect(() => {
    const intervals = [600, 1200, 1800];
    const timers = intervals.map((delay, index) =>
      setTimeout(() => {
        setCurrentStep((s) => Math.max(s, index + 1));
      }, delay)
    );

    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="card-shell analyzing-shell">
      <div className="analyzing-content">
        {/* Animated Sound Wave Radar */}
        <div className="radar-container">
          <div className="radar-ripple r1" />
          <div className="radar-ripple r2" />
          <div className="radar-ripple r3" />
          <div className="radar-core">
            <Loader2 size={36} className="spin-icon text-primary" />
          </div>
        </div>

        <h2 className="analyzing-heading">{t.analyzingHeading}</h2>
        <p className="analyzing-sub">{t.analyzingSubtitle}</p>

        {/* Step progression indicators */}
        <div className="step-pipeline">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            const isDone = currentStep > idx;
            const isCurrent = currentStep === idx;

            return (
              <div
                key={idx}
                className={`pipeline-step ${isDone ? "step-done" : ""} ${isCurrent ? "step-active" : ""}`}
              >
                <div className="step-badge">
                  {isDone ? "✓" : <Icon size={14} />}
                </div>
                <span className="step-label">{step.label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
