"""
model.py
========

The model of the paper

    "Transparency, Confidence and Bank Runs:
     A Simple Model with Treasury Implications"   (Marco Fantin, 2026)

This file contains ONLY the economics: no plots, no sliders.
Each function corresponds to one equation of Section 3 of the paper,
so the code can be read side by side with the text.

    Paper                                          Function in this file
    -----------------------------------------------------------------------
    Eq. (1)  run  <=>  w > ell                     is_run
    Eq. (2)  theta = d - s                         true_soundness
    Eq. (4)  F = omega(tau) * y                    omega, confidence
    Eq. (5)  E[F] = omega(tau) * (d - s)           expected_confidence
    Eq. (6)  w(F) = lam + (1 - lam) * Phi(...)     withdrawals
    Eq. (7)  w(F*) = ell                           critical_confidence

Intuition in one paragraph
--------------------------
Depositors cannot see how sound their bank really is. They combine what
they already believed (the bank's reputation) with what the bank discloses.
Transparency decides how much weight the disclosure gets. Their resulting
estimate is their "confidence". Each depositor leaves if confidence falls
below a personal threshold, and a few depositors leave no matter what.
If too many leave, the bank runs out of liquidity: that is a bank run.

All parameter values used in the paper are illustrative, not calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm  # norm.cdf = Phi, norm.ppf = Phi^{-1}


# =============================================================================
# 1. PARAMETERS
# =============================================================================
# All parameters live in a single object, so that a "scenario" (a bank in a
# given situation) can be passed around as one thing.
#
# frozen=True makes the object read-only: to change a parameter we create a
# new scenario with .with_(...) instead of modifying the old one. This avoids
# accidentally changing a scenario that is used elsewhere.
# =============================================================================
@dataclass(frozen=True)
class Params:
    """A scenario of the model. See Table 1 of the paper."""

    # --- The bank and the crisis -------------------------------------------
    tau: float = 1.0     # Transparency: precision of the bank's disclosure.
                         #   0 = disclosure says nothing, large = reveals all.
    s: float = 0.2       # Crisis severity: how much the shock erodes the
                         #   bank's true soundness.
    d: float = 0.5       # Pre-crisis margin: how much sounder the bank was
                         #   than its reputation before the crisis.
    ell: float = 0.30    # Liquidity buffer: share of deposits the bank can
                         #   pay out immediately (0.30 = 30% of deposits).

    # --- The depositors -----------------------------------------------------
    lam: float = 0.05    # Share of non-rational depositors, who withdraw
                         #   whatever happens (panic, imitation).
    cbar: float = -0.5   # Average withdrawal threshold: how demanding
                         #   depositors are on average (higher = leave sooner).
    sigma: float = 0.5   # Heterogeneity: how much thresholds differ across
                         #   depositors (low = everyone reacts alike).

    def __post_init__(self) -> None:
        """Check that every parameter lies in its admissible range."""
        if self.tau < 0:
            raise ValueError("tau (transparency) must be >= 0.")
        if self.s < 0:
            raise ValueError("s (crisis severity) must be >= 0.")
        if not 0 < self.ell < 1:
            raise ValueError("ell (liquidity buffer) must be between 0 and 1.")
        if not 0 <= self.lam < 1:
            raise ValueError("lam (non-rational share) must be in [0, 1).")
        if self.sigma <= 0:
            raise ValueError("sigma (heterogeneity) must be > 0.")

    def with_(self, **changes: float) -> "Params":
        """Return a copy of this scenario with some parameters changed.

        Example:  severe = base.with_(s=1.0)
        """
        return replace(self, **changes)


# =============================================================================
# 2. FUNDAMENTALS AND CONFIDENCE  (Section 3.2 of the paper)
# =============================================================================

def true_soundness(p: Params) -> float:
    """Eq. (2): theta = d - s.

    The bank's true soundness: its pre-crisis margin minus the damage done
    by the crisis. Depositors never observe this number directly.
    """
    return p.d - p.s


def omega(tau):
    """Weight that depositors put on the bank's disclosure.

        omega = tau / (1 + tau)

    It comes from Bayes' rule: the precision of the disclosure (tau) divided
    by the total precision of the information available (reputation has
    precision 1, disclosure has precision tau).

        tau = 0    ->  omega = 0     (only reputation matters)
        tau = 1    ->  omega = 0.5   (half and half)
        tau -> inf ->  omega -> 1    (only the disclosure matters)

    Works with a single number or with a whole array of values.
    """
    tau = np.asarray(tau, dtype=float)
    return tau / (1.0 + tau)


def confidence(y, tau: float):
    """Eq. (4): confidence F = omega(tau) * y.

    Depositors' best estimate of the bank's soundness after seeing the
    disclosure y. Reputation is normalised to zero, so confidence is simply
    the disclosure scaled by the weight depositors give it.
    """
    return omega(tau) * np.asarray(y, dtype=float)


def expected_confidence(p: Params, tau=None):
    """Eq. (5): expected confidence E[F] = omega(tau) * (d - s).

    On average the disclosure tells the truth (its noise has mean zero), so
    expected confidence is the true soundness scaled by omega.

    By default it uses the transparency of the scenario (p.tau). Passing an
    array of values for `tau` gives the whole curve at once, which is what
    we need to plot confidence against transparency.
    """
    if tau is None:
        tau = p.tau
    return omega(tau) * true_soundness(p)


def marginal_effect_of_transparency(p: Params) -> float:
    """Eq. (5), first derivative: d E[F] / d tau = (d - s) / (1 + tau)^2.

    The key result of the paper. The denominator is always positive, so the
    sign depends only on (d - s):
        crisis milder than the margin (s < d)  ->  transparency HELPS
        crisis worse than the margin  (s > d)  ->  transparency HURTS
    """
    return true_soundness(p) / (1.0 + p.tau) ** 2


# =============================================================================
# 3. DEPOSITORS  (Section 3.3 of the paper)
# =============================================================================

def withdrawals(F, p: Params):
    """Eq. (6): share of depositors withdrawing when confidence equals F.

        w(F) = lam + (1 - lam) * Phi( (cbar - F) / sigma )

    Reading the formula piece by piece:
      * lam                      non-rational depositors: they always leave.
      * (1 - lam)                the rational depositors ...
      * Phi((cbar - F) / sigma)  ... of whom leaves the fraction whose
                                 personal threshold is above current
                                 confidence (thresholds ~ Normal(cbar, sigma^2)).

    When confidence F falls, more thresholds are above it, so w rises.
    Works with a single F or with an array of values (to draw the curve).
    """
    F = np.asarray(F, dtype=float)
    rational_share_leaving = norm.cdf((p.cbar - F) / p.sigma)
    return p.lam + (1.0 - p.lam) * rational_share_leaving


# =============================================================================
# 4. RUN CONDITION  (Section 3.4 of the paper)
# =============================================================================

def critical_confidence(p: Params) -> float:
    """Eq. (7): the critical confidence F* below which the bank fails.

    In the paper F* is defined implicitly by  w(F*) = ell, i.e. "the level
    of confidence at which withdrawals exactly use up the liquidity buffer".
    Because w is monotone in F, the equation can be solved exactly:

        Phi( (cbar - F*) / sigma ) = (ell - lam) / (1 - lam)
        F* = cbar - sigma * Phi^{-1}( (ell - lam) / (1 - lam) )

    (ell - lam) / (1 - lam) has a clear meaning: the share of RATIONAL
    depositors the bank can afford to lose, after paying the non-rational
    ones.

    Special case: if non-rational depositors alone exceed the buffer
    (lam >= ell), the bank fails whatever anyone believes. We return
    +infinity: every level of confidence is "below the threshold".
    """
    if p.lam >= p.ell:
        return np.inf

    share_rational_bank_can_lose = (p.ell - p.lam) / (1.0 - p.lam)
    return p.cbar - p.sigma * norm.ppf(share_rational_bank_can_lose)


def is_run(p: Params) -> bool:
    """Eq. (1): is there a run in this scenario?

    A run occurs when expected confidence is below the critical level F*,
    i.e. when too many depositors leave for the buffer to cover them.
    """
    return bool(expected_confidence(p) < critical_confidence(p))


def critical_transparency(p: Params) -> float:
    """Level of transparency at which the bank crosses into the run region.

    Solves  omega(tau) * (d - s) = F*  for tau.

    A crossing exists only in a severe crisis (d - s < 0): there, confidence
    falls as transparency rises, and at some point it hits F*. Beyond that
    level, disclosing more pushes the bank into a run.
    Returns NaN (not a number) when no crossing exists.
    """
    theta = true_soundness(p)
    F_star = critical_confidence(p)

    # No crossing if the bank is always in a run, or if soundness is zero.
    if not np.isfinite(F_star) or theta == 0:
        return np.nan

    # omega(tau*) must equal F* / theta, and omega is always between 0 and 1.
    omega_star = F_star / theta
    if not 0 < omega_star < 1:
        return np.nan

    # Invert omega = tau / (1 + tau)  ->  tau = omega / (1 - omega).
    return omega_star / (1.0 - omega_star)


# =============================================================================
# 5. SUMMARY OF A SCENARIO
# =============================================================================

def summary(p: Params) -> dict:
    """All the key numbers of a scenario, in one dictionary."""
    EF = float(expected_confidence(p))
    return {
        "theta (true soundness)": true_soundness(p),
        "omega (weight on disclosure)": float(omega(p.tau)),
        "E[F] (expected confidence)": EF,
        "dE[F]/dtau (effect of transparency)": marginal_effect_of_transparency(p),
        "F* (critical confidence)": critical_confidence(p),
        "w (share withdrawing)": float(withdrawals(EF, p)),
        "run": is_run(p),
        "tau* (critical transparency)": critical_transparency(p),
    }


# =============================================================================
# Quick check: running `python model.py` prints two scenarios.
# =============================================================================
if __name__ == "__main__":
    base = Params()
    scenarios = {
        "Mild crisis (s = 0.2)": base,
        "Severe crisis (s = 1.0)": base.with_(s=1.0),
    }

    for name, p in scenarios.items():
        print(f"\n{name}")
        print("-" * len(name))
        for label, value in summary(p).items():
            if isinstance(value, bool):
                print(f"  {label:<38} {value}")
            else:
                print(f"  {label:<38} {value:+.3f}")
