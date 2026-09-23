"""
plotting.py
===========

Everything related to drawing the model. It is shared by:

    explorer.py              the interactive dashboard (desktop window)
    bank_run_explorer.ipynb  the interactive notebook (Jupyter / Binder)

so that both show exactly the same charts.

The dashboard has three panels, one for each step of the model:

    (A) Confidence   How expected confidence changes with transparency,
                     compared with the run threshold F*.       [Eqs. 4-5]
    (B) Withdrawals  How many depositors leave as confidence falls,
                     compared with the liquidity buffer.        [Eqs. 6-7]
    (C) Run map      Which combinations of transparency and crisis
                     severity lead to a run.                     [Eq. 1]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from model import (Params, critical_confidence, expected_confidence,
                   omega, withdrawals)


# =============================================================================
# 1. STYLE
# =============================================================================
# One accent colour (red) marks everything related to a run: the threshold,
# the buffer, the run region. Blue marks a safe bank. Everything else is black
# or grey, so that the figures remain readable when printed.
# =============================================================================
RED = "#B22222"          # runs, thresholds, buffer
BLUE = "#2E6F9E"         # safe bank
LIGHT_RED = "#F6D5D5"    # run region in panel (C)
LIGHT_BLUE = "#EEF3F8"   # safe region in panel (C)


def apply_style() -> None:
    """Academic look: serif font, LaTeX-style maths, no top/right borders."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",       # Computer Modern, as in LaTeX
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# =============================================================================
# 2. GRIDS
# =============================================================================
# The ranges of values over which the curves are drawn.
# =============================================================================
TAU_GRID = np.linspace(0.0, 6.0, 400)    # transparency, panels (A) and (C)
S_GRID = np.linspace(0.0, 2.0, 300)      # crisis severity, panel (C)
F_GRID = np.linspace(-2.0, 1.5, 600)     # confidence, panel (B)


def percent_axis(ax) -> None:
    """Show the y-axis as percentages (0.3 -> 30%)."""
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))


# =============================================================================
# 3. PANEL (A): CONFIDENCE VS TRANSPARENCY
# =============================================================================
def draw_confidence_panel(ax, p: Params, colour: str) -> None:
    """Expected confidence as transparency rises, with the run threshold.

    If the black curve ends up in the shaded red area, depositors'
    confidence is below F* and the bank faces a run.
    """
    F_star = critical_confidence(p)
    threshold_exists = np.isfinite(F_star)

    # The curve E[F](tau) for the current crisis severity.
    curve = expected_confidence(p, TAU_GRID)
    ax.plot(TAU_GRID, curve, color="black", lw=2)

    # The run threshold and the run region below it.
    if threshold_exists:
        ax.axhline(F_star, color=RED, lw=1.3, label=r"Run threshold $F^*$")
        ax.axhspan(-50, F_star, color=RED, alpha=0.07, lw=0)

    # Reference line at zero (= the bank's reputation).
    ax.axhline(0, color="grey", lw=0.6, ls=":")

    # The current bank: a dot at the chosen transparency.
    ax.axvline(p.tau, color="grey", lw=0.8, ls=":")
    ax.plot(p.tau, expected_confidence(p), "o", color=colour, ms=8, zorder=4)

    # Vertical range: always show the whole curve and the threshold.
    reference = F_star if threshold_exists else 0.0
    ax.set_ylim(min(curve.min(), reference) - 0.15,
                max(curve.max(), reference) + 0.15)
    ax.set_xlim(TAU_GRID[0], TAU_GRID[-1])

    ax.set_title("(A) Confidence vs transparency", loc="left")
    ax.set_xlabel(r"Transparency $\tau$")
    ax.set_ylabel(r"Expected confidence $\mathbb{E}[F]$")
    if threshold_exists:
        ax.legend(frameon=False, loc="upper right")


# =============================================================================
# 4. PANEL (B): WITHDRAWALS VS CONFIDENCE
# =============================================================================
def draw_withdrawal_panel(ax, p: Params, colour: str) -> None:
    """Share of depositors withdrawing as confidence falls.

    Read it from right to left: as confidence falls, withdrawals rise.
    Where the black curve crosses the red line, withdrawals exhaust the
    liquidity buffer: that point is F*.
    """
    F_star = critical_confidence(p)
    EF = float(expected_confidence(p))

    # The withdrawal curve w(F).
    ax.plot(F_GRID, withdrawals(F_GRID, p), color="black", lw=2)

    # The liquidity buffer and the floor set by non-rational depositors.
    ax.axhline(p.ell, color=RED, lw=1.3, label=r"Liquidity buffer $\ell$")
    ax.axhline(p.lam, color="grey", lw=0.6, ls=":",
               label=r"Non-rational share $\lambda$")

    # The critical confidence F* (dashed vertical line).
    if np.isfinite(F_star):
        ax.axvline(F_star, color=RED, lw=0.8, ls="--")

    # The current bank: a dot at its expected confidence.
    ax.plot(EF, withdrawals(EF, p), "o", color=colour, ms=8, zorder=4)

    ax.set_xlim(F_GRID[0], F_GRID[-1])
    ax.set_ylim(0, 1.02)
    percent_axis(ax)

    ax.set_title("(B) Withdrawals vs confidence", loc="left")
    ax.set_xlabel(r"Confidence $F$  (falling $\leftarrow$)")
    ax.set_ylabel(r"Share withdrawing $w$")
    ax.legend(frameon=False, loc="upper right")


# =============================================================================
# 5. PANEL (C): RUN MAP
# =============================================================================
def draw_run_map(ax, p: Params, colour: str) -> None:
    """Map of all (transparency, crisis severity) pairs: run or no run.

    Every point of the map is a possible bank with the same depositors and
    buffer as the current one, but different transparency and crisis.
    Red = run, blue = no run. The dot is the current bank.
    """
    F_star = critical_confidence(p)

    # Build a grid of (tau, s) pairs and check the run condition in each:
    #     run  <=>  omega(tau) * (d - s) < F*
    T, S = np.meshgrid(TAU_GRID, S_GRID)
    run_region = omega(T) * (p.d - S) < F_star

    # Colour the grid: 0 = no run (light blue), 1 = run (light red).
    ax.imshow(run_region, origin="lower", aspect="auto",
              extent=[TAU_GRID[0], TAU_GRID[-1], S_GRID[0], S_GRID[-1]],
              cmap=plt.matplotlib.colors.ListedColormap([LIGHT_BLUE,
                                                         LIGHT_RED]),
              vmin=0, vmax=1)

    # The boundary between the two regions, solving the run condition for s:
    #     s = d - F* / omega(tau)
    if np.isfinite(F_star):
        with np.errstate(divide="ignore", invalid="ignore"):
            boundary = p.d - F_star / omega(TAU_GRID)
        ax.plot(TAU_GRID, boundary, color=RED, lw=1.5, label="Run boundary")

    # Crisis equal to the pre-crisis margin: above this line, transparency
    # starts to hurt confidence instead of helping it.
    ax.axhline(p.d, color="grey", lw=0.8, ls=":", label=r"$s = d$")

    # The current bank.
    ax.plot(p.tau, p.s, "o", color=colour, mec="black", ms=9, zorder=4)

    # Region labels.
    ax.text(0.97, 0.95, "RUN", transform=ax.transAxes, ha="right", va="top",
            color=RED, fontweight="bold")
    ax.text(0.97, 0.05, "NO RUN", transform=ax.transAxes, ha="right",
            va="bottom", color=BLUE, fontweight="bold")

    ax.set_xlim(TAU_GRID[0], TAU_GRID[-1])
    ax.set_ylim(S_GRID[0], S_GRID[-1])
    ax.set_title("(C) Run map", loc="left")
    ax.set_xlabel(r"Transparency $\tau$")
    ax.set_ylabel(r"Crisis severity $s$")
    ax.legend(frameon=False, loc="upper center", fontsize=9)


# =============================================================================
# 6. THE WHOLE DASHBOARD
# =============================================================================
def verdict(p: Params) -> tuple[str, str]:
    """Plain-language verdict for a scenario, and the colour to show it in."""
    F_star = critical_confidence(p)
    EF = float(expected_confidence(p))
    w = float(withdrawals(EF, p))
    F_text = f"{F_star:+.3f}" if np.isfinite(F_star) else "+inf"

    if not np.isfinite(F_star):
        message, colour = ("RUN: non-rational depositors alone exhaust "
                           "the buffer"), RED
    elif EF < F_star:
        message, colour = "RUN: withdrawals exceed the liquidity buffer", RED
    else:
        message, colour = "NO RUN: the liquidity buffer holds", BLUE

    numbers = (f"E[F] = {EF:+.3f}    F* = {F_text}    "
               f"withdrawals = {w:.1%}    buffer = {p.ell:.0%}")
    return f"{message}\n{numbers}", colour


def draw_dashboard(axes, p: Params) -> tuple[str, str]:
    """Draw the three panels on the given axes and return the verdict.

    `axes` is a list of three matplotlib axes: (A), (B), (C).
    """
    text, colour = verdict(p)
    ax_a, ax_b, ax_c = axes
    for ax in axes:
        ax.clear()  # start from an empty panel at every update
    draw_confidence_panel(ax_a, p, colour)
    draw_withdrawal_panel(ax_b, p, colour)
    draw_run_map(ax_c, p, colour)
    return text, colour
