import React, { useState } from "react";
import { Activity, Stethoscope, FileText, ChevronRight, Globe, Shield, X, CheckCircle, ArrowRight } from "lucide-react";

export default function LandingScreen({ onGetStarted, onLogIn, t, currentLang, onToggleLang }) {
  const [activeFeatureModal, setActiveFeatureModal] = useState(null);

  const featureDetails = {
    respiratory: {
      icon: Activity,
      image: "/images/hero_pasture_flock.jpg",
      title: t.featureRespiratoryTitle || "Respiratory Health",
      tagline: t.featureRespiratoryDesc || "Early rale & wheeze detection",
      details: [
        "Continuous 5-second acoustic window scanning across coop enclosures.",
        "Deep convolutional neural network trained on over 20,000+ empirical poultry vocalizations.",
        "Detects sub-audible bronchial rattling and inspiratory stridor before visible flock lethargy.",
      ],
      badge: "Real-time AI Saliency",
    },
    differential: {
      icon: Stethoscope,
      image: "/images/broiler_coop_flock.png",
      title: t.featureDifferentialTitle || "Disease Differential",
      tagline: t.featureDifferentialDesc || "IBV, CRD & Coryza matching",
      details: [
        "Differential pattern matching for Infectious Bronchitis (IBV) and Chronic Respiratory Disease (CRD).",
        "Acoustic screening for Infectious Coryza, Newcastle Disease (NDV), and Aspergillosis.",
        "Generates interactive coop symptom verification checklists for flock managers.",
      ],
      badge: "5 Avian Pathogens",
    },
    reports: {
      icon: FileText,
      image: "/images/vet_biosecurity_inspection.jpg",
      title: t.featureVetReportsTitle || "Veterinary Reports",
      tagline: t.featureVetReportsDesc || "Clinical export for veterinarians",
      details: [
        "One-click printable clinical summary formatted for licensed avian veterinarians.",
        "Full acoustic biomarker breakdown (Tracheal Rale Power %, Spectral Centroid, Event Density).",
        "Includes farm biosecurity containment guidelines and local poultry clinic locator.",
      ],
      badge: "Printable Clinical PDF",
    },
  };

  function handleCardClick(featureKey) {
    setActiveFeatureModal(featureKey);
  }

  return (
    <div className="landing-screen-shell">
      {/* Top Bar with Logo Emblem & Language Toggle */}
      <header className="landing-header">
        <div className="lettermark-brand">
          <img
            src="/images/flockcare_logo.png"
            alt="FlockCare Logo"
            className="brand-logo-img"
          />
          <span className="serif-brand">FlockCare</span>
        </div>
        <button
          className="lang-pill-btn"
          onClick={() => onToggleLang(currentLang === "en" ? "hi" : "en")}
          title="Toggle Language"
        >
          <Globe size={13} />
          <span>{currentLang === "en" ? "हिंदी" : "EN"}</span>
        </button>
      </header>

      {/* Main Hero Content */}
      <main className="landing-content">
        <div className="hero-typography-group">
          <h1 className="hero-serif-title">
            {t.landingHeroPrefix}
            <span className="hero-serif-highlight">{t.landingHeroHighlight}</span>
          </h1>
          <p className="hero-serif-subtitle">{t.landingSubtitle}</p>
        </div>

        {/* Feature Cards Grid (Inspired by Hers category cards with Photography) */}
        <div className="landing-photo-grid">
          {/* Card 1: Respiratory Health (Pasture Flock) */}
          <div
            className="landing-photo-card"
            onClick={() => handleCardClick("respiratory")}
            role="button"
            tabIndex={0}
          >
            <div className="photo-card-media-wrapper">
              <img
                src="/images/hero_pasture_flock.jpg"
                alt="Flock pasture monitoring"
                className="photo-card-img"
                loading="eager"
              />
              <div className="photo-card-badge">Acoustic AI</div>
            </div>
            <div className="photo-card-info">
              <span className="photo-card-category">{t.featureRespiratoryTitle || "Respiratory Health"}</span>
              <div className="photo-card-arrow-row">
                <span className="photo-card-tagline">{t.featureRespiratoryDesc || "Early rale & wheeze detection"}</span>
                <ChevronRight size={15} className="photo-arrow-icon" />
              </div>
            </div>
          </div>

          {/* Card 2: Disease Differential (Commercial Broiler Flock) */}
          <div
            className="landing-photo-card"
            onClick={() => handleCardClick("differential")}
            role="button"
            tabIndex={0}
          >
            <div className="photo-card-media-wrapper">
              <img
                src="/images/broiler_coop_flock.png"
                alt="Commercial coop flock health"
                className="photo-card-img"
                loading="eager"
              />
              <div className="photo-card-badge">Differentials</div>
            </div>
            <div className="photo-card-info">
              <span className="photo-card-category">{t.featureDifferentialTitle || "Disease Differential"}</span>
              <div className="photo-card-arrow-row">
                <span className="photo-card-tagline">{t.featureDifferentialDesc || "IBV, CRD & Coryza matching"}</span>
                <ChevronRight size={15} className="photo-arrow-icon" />
              </div>
            </div>
          </div>

          {/* Card 3: Veterinary Biosecurity Reports (Wide Banner) */}
          <div
            className="landing-photo-card photo-card-wide"
            onClick={() => handleCardClick("reports")}
            role="button"
            tabIndex={0}
          >
            <div className="photo-card-media-wrapper wide-media">
              <img
                src="/images/vet_biosecurity_inspection.jpg"
                alt="Veterinary biosecurity assessment"
                className="photo-card-img"
                loading="eager"
              />
              <div className="photo-card-badge">Clinical Vet Plan</div>
            </div>
            <div className="photo-card-info">
              <span className="photo-card-category">{t.featureVetReportsTitle || "Veterinary Reports"}</span>
              <div className="photo-card-arrow-row">
                <span className="photo-card-tagline">{t.featureVetReportsDesc || "Clinical export for veterinarians"}</span>
                <ChevronRight size={15} className="photo-arrow-icon" />
              </div>
            </div>
          </div>
        </div>

        {/* Action Button Stack */}
        <div className="landing-cta-stack">
          <button className="btn-pill-solid" onClick={onGetStarted}>
            {t.getStarted}
          </button>

          <button className="btn-pill-outline" onClick={onLogIn}>
            {t.logIn}
          </button>

          <div className="landing-account-prompt">
            <span>{t.newToApp} </span>
            <button className="text-link-btn" onClick={onLogIn}>
              {t.createAccount}
            </button>
          </div>
        </div>
      </main>

      {/* Feature Preview Modal */}
      {activeFeatureModal && featureDetails[activeFeatureModal] && (
        <div className="modal-overlay" onClick={() => setActiveFeatureModal(null)}>
          <div className="modal-content feature-info-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                {React.createElement(featureDetails[activeFeatureModal].icon, { size: 20 })}
                <h3 className="modal-title">{featureDetails[activeFeatureModal].title}</h3>
              </div>
              <button
                className="icon-btn-close"
                onClick={() => setActiveFeatureModal(null)}
                aria-label="Close feature details"
              >
                <X size={18} />
              </button>
            </div>

            <div className="modal-body">
              <div className="feature-modal-img-container">
                <img
                  src={featureDetails[activeFeatureModal].image}
                  alt={featureDetails[activeFeatureModal].title}
                  className="feature-modal-img"
                />
              </div>

              <div className="feature-modal-banner">
                <span className="feature-modal-pill">{featureDetails[activeFeatureModal].badge}</span>
                <p className="feature-modal-tagline">{featureDetails[activeFeatureModal].tagline}</p>
              </div>

              <div className="feature-points-list">
                {featureDetails[activeFeatureModal].details.map((point, pIdx) => (
                  <div key={pIdx} className="feature-point-item">
                    <CheckCircle size={15} className="text-healthy flex-shrink-0" />
                    <span>{point}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="modal-footer">
              <button
                className="btn-pill-solid"
                onClick={() => {
                  setActiveFeatureModal(null);
                  onLogIn();
                }}
              >
                <span>Sign in to Access {featureDetails[activeFeatureModal].title}</span>
                <ArrowRight size={15} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Trust Badge Footer */}
      <footer className="landing-footer">
        <Shield size={12} className="text-muted" />
        <span className="trust-text">{t.trustedBadge}</span>
      </footer>
    </div>
  );
}
