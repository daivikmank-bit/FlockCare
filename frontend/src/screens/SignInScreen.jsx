import React, { useState } from "react";
import { ArrowLeft, CheckSquare, Square, Lock, Mail, UserCheck, AlertCircle, Sparkles } from "lucide-react";

export default function SignInScreen({ onSignInSuccess, onBackToLanding, onGuestContinue, t }) {
  const [isSignUpMode, setIsSignUpMode] = useState(false);
  const [farmName, setFarmName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(true);
  const [formError, setFormError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  function handleSubmit(e) {
    e.preventDefault();
    setFormError(null);

    // Validation
    const nameToUse = farmName.trim();
    if (!nameToUse) {
      setFormError("Please enter your farm name or identifier.");
      return;
    }

    if (!password.trim() || password.length < 4) {
      setFormError("Please enter a valid password or passcode (min 4 characters).");
      return;
    }

    if (isSignUpMode && confirmPassword !== password) {
      setFormError("Passwords do not match. Please verify.");
      return;
    }

    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      onSignInSuccess(nameToUse, rememberMe);
    }, 400);
  }

  function toggleMode() {
    setIsSignUpMode(!isSignUpMode);
    setFormError(null);
  }

  return (
    <div className="signin-screen-shell">
      {/* Top Header with Back Navigation & Logo */}
      <header className="signin-header">
        <button
          className="back-icon-btn"
          onClick={onBackToLanding}
          aria-label="Back to starting page"
        >
          <ArrowLeft size={18} />
        </button>
        <div className="signin-brand-center">
          <img
            src="/images/flockcare_logo.png"
            alt="FlockCare"
            className="brand-logo-xs"
          />
          <span className="serif-brand-sm">FlockCare</span>
        </div>
        <div style={{ width: 32 }} /> {/* balance spacer */}
      </header>

      {/* Main Sign-In / Sign-Up Form */}
      <main className="signin-content">
        <div className="signin-photo-header">
          <div className="signin-thumb-wrapper">
            <img
              src="/images/farmer_holding_hen.png"
              alt="FlockCare poultry wellness"
              className="signin-thumb-img"
            />
          </div>
          <div className="signin-typography-group">
            <h1 className="signin-serif-title">
              {isSignUpMode ? "Create Farm Account" : t.signInTitle}
            </h1>
            <p className="signin-serif-subtitle">
              {isSignUpMode
                ? "Register your poultry facility for automated acoustic surveillance."
                : t.signInSubtitle}
            </p>
          </div>
        </div>

        {formError && (
          <div className="banner error-banner">
            <AlertCircle size={16} className="banner-icon" />
            <div className="banner-text">{formError}</div>
          </div>
        )}

        <form className="signin-form" onSubmit={handleSubmit}>
          {/* Farm Name Input */}
          <div className="input-field-group">
            <label className="input-label" htmlFor="farm-id-input">
              {isSignUpMode ? "Farm or Coop Name" : t.emailOrFarmLabel}
            </label>
            <div className="input-wrapper">
              <input
                id="farm-id-input"
                type="text"
                className="text-input"
                placeholder={isSignUpMode ? "e.g. Highland Layer Farm" : t.emailPlaceholder}
                value={farmName}
                onChange={(e) => {
                  setFarmName(e.target.value);
                  if (formError) setFormError(null);
                }}
                autoComplete="organization"
                required
              />
            </div>
          </div>

          {/* Email / Operator ID in Sign Up mode */}
          {isSignUpMode && (
            <div className="input-field-group">
              <label className="input-label" htmlFor="email-input">
                Operator Email or Phone
              </label>
              <div className="input-wrapper">
                <input
                  id="email-input"
                  type="text"
                  className="text-input"
                  placeholder="e.g. manager@highlandfarm.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                />
              </div>
            </div>
          )}

          {/* Password Input */}
          <div className="input-field-group">
            <div className="input-label-row">
              <label className="input-label" htmlFor="password-input">
                {t.passwordLabel}
              </label>
              {!isSignUpMode && (
                <button
                  type="button"
                  className="forgot-pass-btn"
                  onClick={() => alert("Password reset instructions sent to registered contact.")}
                >
                  {t.forgotPassword}
                </button>
              )}
            </div>
            <div className="input-wrapper">
              <input
                id="password-input"
                type="password"
                className="text-input"
                placeholder={t.passwordPlaceholder}
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  if (formError) setFormError(null);
                }}
                autoComplete={isSignUpMode ? "new-password" : "current-password"}
                required
              />
            </div>
          </div>

          {/* Confirm Password in Sign Up mode */}
          {isSignUpMode && (
            <div className="input-field-group">
              <label className="input-label" htmlFor="confirm-password-input">
                Confirm Password
              </label>
              <div className="input-wrapper">
                <input
                  id="confirm-password-input"
                  type="password"
                  className="text-input"
                  placeholder="Re-enter password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </div>
            </div>
          )}

          {/* Remember Device Toggle */}
          <div
            className="remember-device-row"
            onClick={() => setRememberMe(!rememberMe)}
            role="button"
            tabIndex={0}
          >
            {rememberMe ? (
              <CheckSquare size={16} className="text-dark" />
            ) : (
              <Square size={16} className="text-muted" />
            )}
            <span className="remember-text">{t.rememberDevice}</span>
          </div>

          {/* Submit & Guest Button Stack */}
          <div className="signin-cta-stack">
            <button type="submit" className="btn-pill-solid" disabled={isSubmitting}>
              {isSubmitting
                ? "Processing…"
                : isSignUpMode
                ? "Create Account & Start"
                : t.signInBtn}
            </button>

            <button
              type="button"
              className="btn-pill-outline"
              onClick={onGuestContinue}
            >
              <UserCheck size={15} />
              <span>{t.continueAsGuest}</span>
            </button>
          </div>
        </form>

        {/* Toggle between Sign In & Sign Up */}
        <div className="signin-footer-prompt">
          <span>{isSignUpMode ? "Already have a farm account?" : t.noAccountPrompt} </span>
          <button className="text-link-btn" onClick={toggleMode}>
            {isSignUpMode ? "Sign in instead" : t.signUpLink}
          </button>
        </div>
      </main>
    </div>
  );
}
