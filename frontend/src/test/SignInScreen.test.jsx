import React, { act } from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import SignInScreen from "../screens/SignInScreen";
import en from "../i18n/en";

describe("SignInScreen Component", () => {
  it("renders welcome back title, input labels, and bypass button", () => {
    const onSignInSuccess = vi.fn();
    const onBackToLanding = vi.fn();
    const onGuestContinue = vi.fn();

    render(
      <SignInScreen
        onSignInSuccess={onSignInSuccess}
        onBackToLanding={onBackToLanding}
        onGuestContinue={onGuestContinue}
        t={en}
      />
    );

    expect(screen.getByText("Welcome back")).toBeInTheDocument();
    expect(screen.getByLabelText(/Farm identifier or Email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Password \/ Passcode/i)).toBeInTheDocument();
    expect(screen.getByText("Continue as Guest Farmer")).toBeInTheDocument();
  });

  it("calls onBackToLanding when back icon button is clicked", () => {
    const onSignInSuccess = vi.fn();
    const onBackToLanding = vi.fn();
    const onGuestContinue = vi.fn();

    render(
      <SignInScreen
        onSignInSuccess={onSignInSuccess}
        onBackToLanding={onBackToLanding}
        onGuestContinue={onGuestContinue}
        t={en}
      />
    );

    const backBtn = screen.getByRole("button", { name: /Back to starting page/i });
    fireEvent.click(backBtn);
    expect(onBackToLanding).toHaveBeenCalledTimes(1);
  });

  it("calls onGuestContinue when 'Continue as Guest Farmer' is clicked", () => {
    const onSignInSuccess = vi.fn();
    const onBackToLanding = vi.fn();
    const onGuestContinue = vi.fn();

    render(
      <SignInScreen
        onSignInSuccess={onSignInSuccess}
        onBackToLanding={onBackToLanding}
        onGuestContinue={onGuestContinue}
        t={en}
      />
    );

    const guestBtn = screen.getByText("Continue as Guest Farmer");
    fireEvent.click(guestBtn);
    expect(onGuestContinue).toHaveBeenCalledTimes(1);
  });

  it("validates empty inputs and submits successfully when filled", async () => {
    vi.useFakeTimers();
    const onSignInSuccess = vi.fn();
    const onBackToLanding = vi.fn();
    const onGuestContinue = vi.fn();

    render(
      <SignInScreen
        onSignInSuccess={onSignInSuccess}
        onBackToLanding={onBackToLanding}
        onGuestContinue={onGuestContinue}
        t={en}
      />
    );

    const farmInput = screen.getByLabelText(/Farm identifier or Email/i);
    const passwordInput = screen.getByLabelText(/Password \/ Passcode/i);

    // Enter credentials
    fireEvent.change(farmInput, { target: { value: "Sunny Pastures Coop" } });
    fireEvent.change(passwordInput, { target: { value: "pass1234" } });

    const submitBtn = screen.getByRole("button", { name: "Sign in" });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    await act(async () => {
      vi.advanceTimersByTime(500);
    });

    expect(onSignInSuccess).toHaveBeenCalledWith("Sunny Pastures Coop");
    vi.useRealTimers();
  });
});
