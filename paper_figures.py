"""
paper_figures.py
================

Reproduces the two figures of Section 4 (Results) of the paper and saves
them in the folder `figures/`, both as:

    PDF  vector format, stays sharp at any zoom -> use it in LaTeX/Overleaf
    PNG  image format                            -> use it in slides or README

    Figure 1  Transparency cuts both ways
    Figure 2  Heterogeneity: probability versus severity of a run

How to run it:   python paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import (Params, critical_confidence, critical_transparency,
                   expected_confidence, withdrawals)
from plotting import RED, apply_style, percent_axis


# =============================================================================
# SETTINGS
# =============================================================================
# The baseline scenario of the paper (illustrative, not calibrated).
BASELINE = Params(tau=1.0, s=0.2, d=0.5, ell=0.30,
                  lam=0.05, cbar=-0.5, sigma=0.5)

FIGURE_SIZE = (6.5, 4.0)      # inches: fits the text width of an A4 paper
OUTPUT_FOLDER = Path("figures")


# =============================================================================
# FIGURE 1: TRANSPARENCY CUTS BOTH WAYS
# =============================================================================
def figure_transparency(p: Params = BASELINE):
    """Expected confidence against transparency, in a mild and a severe crisis.

    Mild crisis (s < d):   confidence RISES with transparency.
    Severe crisis (s > d): confidence FALLS with transparency and, beyond
                           tau*, drops below the run threshold F*.
    """
    tau = np.linspace(0, 6, 500)
    F_star = critical_confidence(p)

    # The two scenarios differ only in crisis severity.
    mild = p.with_(s=0.2)
    severe = p.with_(s=1.0)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # --- The two confidence curves ------------------------------------------
    ax.plot(tau, expected_confidence(mild, tau), color="black", lw=1.8,
            label=rf"Mild crisis ($s={mild.s}<d$)")
    ax.plot(tau, expected_confidence(severe, tau), color="black", lw=1.8,
            ls="--", label=rf"Severe crisis ($s={severe.s}>d$)")

    # --- Run threshold and run region ---------------------------------------
    ax.axhline(F_star, color=RED, lw=1.3)
    ax.axhspan(-10, F_star, color=RED, alpha=0.06, lw=0)
    ax.text(tau[-1], F_star - 0.03, r"Run threshold $F^*$",
            color=RED, ha="right", va="top")

    # Reference line at zero (= the bank's reputation).
    ax.axhline(0, color="grey", lw=0.6, ls=":")

    # --- Critical transparency tau*: where the severe curve crosses F* -----
    tau_star = critical_transparency(severe)
    if np.isfinite(tau_star):
        ax.axvline(tau_star, color="grey", lw=0.8, ls=":")
        ax.annotate(rf"$\tau^* \approx {tau_star:.2f}$",
                    xy=(tau_star, F_star),
                    xytext=(tau_star + 0.35, F_star + 0.12),
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.8))

    # --- Axes ----------------------------------------------------------------
    ax.set_xlim(0, tau[-1])
    ax.set_ylim(-0.55, 0.4)
    ax.set_xlabel(r"Transparency $\tau$")
    ax.set_ylabel(r"Expected confidence $\mathbb{E}[F]$")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE 2: HETEROGENEITY - PROBABILITY VERSUS SEVERITY
# =============================================================================
def figure_heterogeneity(p: Params = BASELINE):
    """Withdrawal curves for a homogeneous and a heterogeneous depositor base.

    Homogeneous base (low sigma):   no one leaves until confidence nears the
                                    common threshold, then almost everyone
                                    leaves at once -> later, but violent.
    Heterogeneous base (high sigma): the most nervous leave early, the rest
                                    follow gradually -> earlier, but slow.
    """
    F = np.linspace(-1.2, 0.4, 600)

    # (sigma, line style, name) for the two depositor bases.
    depositor_bases = [
        (0.1, "--", "Homogeneous base"),
        (0.6, "-", "Heterogeneous base"),
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for sigma, line_style, name in depositor_bases:
        scenario = p.with_(sigma=sigma)

        # The withdrawal curve w(F).
        ax.plot(F, withdrawals(F, scenario), color="black", lw=1.8,
                ls=line_style, label=rf"{name} ($\sigma={sigma}$)")

        # The point where withdrawals exhaust the buffer: (F*, ell).
        F_star = critical_confidence(scenario)
        ax.plot(F_star, scenario.ell, "o", color=RED, ms=5, zorder=3)
        ax.annotate(rf"$F^*={F_star:.2f}$", xy=(F_star, scenario.ell),
                    xytext=(F_star + 0.04, scenario.ell + 0.07),
                    fontsize=9, color=RED)

    # --- Liquidity buffer and non-rational floor ----------------------------
    ax.axhline(p.ell, color=RED, lw=1.3)
    ax.text(F[-1], p.ell + 0.015, r"Liquidity buffer $\ell$", color=RED,
            ha="right", va="bottom")
    ax.axhline(p.lam, color="grey", lw=0.6, ls=":")
    ax.text(F[-1], p.lam + 0.015, r"Non-rational share $\lambda$",
            color="grey", ha="right", va="bottom", fontsize=9)

    # --- Axes ----------------------------------------------------------------
    ax.set_xlim(F[0], F[-1])
    ax.set_ylim(0, 1.02)
    percent_axis(ax)
    ax.set_xlabel(r"Depositor confidence $F$  (falling $\leftarrow$)")
    ax.set_ylabel(r"Share of depositors withdrawing $w$")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


# =============================================================================
# SAVE BOTH FIGURES
# =============================================================================
def save_all(folder: Path = OUTPUT_FOLDER) -> None:
    """Create both figures and save them as PDF and PNG in `folder`."""
    apply_style()
    folder.mkdir(parents=True, exist_ok=True)

    figures = {
        "figure1_transparency": figure_transparency,
        "figure2_heterogeneity": figure_heterogeneity,
    }
    for file_name, make_figure in figures.items():
        fig = make_figure()
        fig.savefig(folder / f"{file_name}.pdf", bbox_inches="tight")
        fig.savefig(folder / f"{file_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {folder / file_name}.pdf and .png")


if __name__ == "__main__":
    save_all()
