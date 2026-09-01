import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import LandingScreen from "../screens/LandingScreen";
import en from "../i18n/en";

describe("LandingScreen Component", () => {
  it("renders classic lettermark, hero typography, and feature preview cards", () => {
    const onGetStarted = vi.fn();
    const onLogIn = vi.fn();
    const onToggleLang = vi.fn();

    render(
      <LandingScreen
        onGetStarted={onGetStarted}
        onLogIn={onLogIn}
        t={en}
        currentLang="en"
        onToggleLang={onToggleLang}
      />
    );

    // Lettermark & Hero
    expect(screen.getByText("FlockCare")).toBeInTheDocument();
    expect(screen.getByText(/Better care designed just for/i)).toBeInTheDocument();
    expect(screen.getByText("your flock")).toBeInTheDocument();

    // Feature Cards
    expect(screen.getByText("Respiratory Health")).toBeInTheDocument();
    expect(screen.getByText("Disease Differential")).toBeInTheDocument();
    expect(screen.getByText("Veterinary Reports")).toBeInTheDocument();

    // Trust badge
    expect(screen.getByText(/Trusted by poultry producers across 50,000\+ birds/i)).toBeInTheDocument();
  });

  it("opens feature preview modal when a feature card is clicked", () => {
    const onGetStarted = vi.fn();
    const onLogIn = vi.fn();
    const onToggleLang = vi.fn();

    render(
      <LandingScreen
        onGetStarted={onGetStarted}
        onLogIn={onLogIn}
        t={en}
        currentLang="en"
        onToggleLang={onToggleLang}
      />
    );

    const respCard = screen.getByText("Respiratory Health");
    fireEvent.click(respCard);

    expect(screen.getByText("Real-time AI Saliency")).toBeInTheDocument();
    expect(screen.getByText(/Continuous 5-second acoustic window scanning/i)).toBeInTheDocument();

    // Click CTA in modal to go to login
    const modalCta = screen.getByText(/Sign in to Access Respiratory Health/i);
    fireEvent.click(modalCta);
    expect(onLogIn).toHaveBeenCalledTimes(1);
  });

  it("calls onGetStarted when 'Get started' button is clicked", () => {
    const onGetStarted = vi.fn();
    const onLogIn = vi.fn();
    const onToggleLang = vi.fn();

    render(
      <LandingScreen
        onGetStarted={onGetStarted}
        onLogIn={onLogIn}
        t={en}
        currentLang="en"
        onToggleLang={onToggleLang}
      />
    );

    const getStartedBtn = screen.getByText("Get started");
    fireEvent.click(getStartedBtn);
    expect(onGetStarted).toHaveBeenCalledTimes(1);
  });

  it("calls onLogIn when 'Log in' button is clicked", () => {
    const onGetStarted = vi.fn();
    const onLogIn = vi.fn();
    const onToggleLang = vi.fn();

    render(
      <LandingScreen
        onGetStarted={onGetStarted}
        onLogIn={onLogIn}
        t={en}
        currentLang="en"
        onToggleLang={onToggleLang}
      />
    );

    const logInBtn = screen.getByText("Log in");
    fireEvent.click(logInBtn);
    expect(onLogIn).toHaveBeenCalledTimes(1);
  });

  it("calls onToggleLang when language pill is clicked", () => {
    const onGetStarted = vi.fn();
    const onLogIn = vi.fn();
    const onToggleLang = vi.fn();

    render(
      <LandingScreen
        onGetStarted={onGetStarted}
        onLogIn={onLogIn}
        t={en}
        currentLang="en"
        onToggleLang={onToggleLang}
      />
    );

    const langBtn = screen.getByTitle("Toggle Language");
    fireEvent.click(langBtn);
    expect(onToggleLang).toHaveBeenCalledWith("hi");
  });
});
