"""
explorer.py
===========

Interactive dashboard of the model in a desktop window, with one slider
for each parameter. The charts are redrawn when you RELEASE a slider.

How to run it (Anaconda)
------------------------
  * Anaconda Prompt:  cd <this folder>   then   python explorer.py
  * Spyder:           Preferences > IPython console > Graphics > Backend:
                      "Automatic" (or type  %matplotlib qt ), then run.

If you prefer Jupyter, use bank_run_explorer.ipynb: it shows exactly the
same charts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from model import Params
from plotting import apply_style, draw_dashboard


# =============================================================================
# 1. STARTING SCENARIO AND SLIDERS
# =============================================================================
# The dashboard opens on the baseline scenario used in the paper.
BASELINE = Params(tau=1.0, s=0.2, d=0.5, ell=0.30,
                  lam=0.05, cbar=-0.5, sigma=0.5)

# One row per slider: (parameter name, label, minimum, maximum, step).
# The names must match the fields of Params in model.py.
SLIDERS = [
    # Left column: the bank and the crisis
    ("tau",   r"Transparency $\tau$",           0.00, 6.00, 0.05),
    ("s",     r"Crisis severity $s$",           0.00, 2.00, 0.01),
    ("d",     r"Pre-crisis margin $d$",        -1.00, 1.00, 0.01),
    ("ell",   r"Liquidity buffer $\ell$",       0.05, 0.95, 0.01),
    # Right column: the depositors
    ("lam",   r"Non-rational share $\lambda$",  0.00, 0.50, 0.01),
    ("cbar",  r"Avg. threshold $\bar c$",      -1.50, 0.50, 0.01),
    ("sigma", r"Heterogeneity $\sigma$",        0.05, 1.50, 0.01),
]


# =============================================================================
# 2. THE DASHBOARD
# =============================================================================
class Explorer:
    """A window with the three panels of the model and the sliders below."""

    def __init__(self) -> None:
        apply_style()
        self.fig = plt.figure(figsize=(15, 8.5))
        self.fig.suptitle("Transparency, Confidence and Bank Runs",
                          fontsize=14, fontweight="bold", y=0.98)

        # --- The three chart panels, side by side in the top half ----------
        #     add_axes([left, bottom, width, height]) in fractions of the window
        self.panels = [
            self.fig.add_axes([0.05, 0.52, 0.27, 0.36]),   # (A)
            self.fig.add_axes([0.38, 0.52, 0.27, 0.36]),   # (B)
            self.fig.add_axes([0.71, 0.52, 0.27, 0.36]),   # (C)
        ]

        # --- The verdict (RUN / NO RUN), centred between charts and sliders --
        self.verdict = self.fig.text(0.5, 0.405, "", fontsize=12,
                                     fontweight="bold", ha="center",
                                     va="center")

        # --- The sliders, in two columns in the bottom half -----------------
        self.sliders = {}
        for i, (name, label, low, high, step) in enumerate(SLIDERS):
            column = 0 if i < 4 else 1
            row = i if i < 4 else i - 4
            x = 0.10 if column == 0 else 0.58
            y = 0.31 - row * 0.055
            slider_axes = self.fig.add_axes([x, y, 0.30, 0.025])

            slider = Slider(slider_axes, label, low, high,
                            valinit=getattr(BASELINE, name),
                            valstep=step, color="0.35")
            # Moving a slider only marks the dashboard as "to be redrawn";
            # the actual redraw happens when the mouse button is released.
            # This keeps the window responsive while dragging.
            slider.on_changed(self._mark_changed)
            self.sliders[name] = slider

        # --- Reset button ----------------------------------------------------
        reset_axes = self.fig.add_axes([0.88, 0.14, 0.08, 0.04])
        self.reset_button = Button(reset_axes, "Reset", color="0.92",
                                   hovercolor="0.85")
        self.reset_button.on_clicked(self._reset)

        # --- Redraw when the mouse is released -------------------------------
        self.changed = False
        self.fig.canvas.mpl_connect("button_release_event",
                                    self._redraw_if_changed)

        self.redraw()

    # -------------------------------------------------------------------------
    def current_scenario(self) -> Params:
        """Read the sliders and build the corresponding scenario."""
        values = {name: float(slider.val)
                  for name, slider in self.sliders.items()}
        return Params(**values)

    def redraw(self) -> None:
        """Redraw the three panels and the verdict for the current sliders."""
        text, colour = draw_dashboard(self.panels, self.current_scenario())
        self.verdict.set_text(text)
        self.verdict.set_color(colour)
        self.fig.canvas.draw_idle()
        self.changed = False

    # --- Event handlers ------------------------------------------------------
    def _mark_changed(self, _value) -> None:
        self.changed = True

    def _redraw_if_changed(self, _event) -> None:
        if self.changed:
            self.redraw()

    def _reset(self, _event) -> None:
        for slider in self.sliders.values():
            slider.reset()
        self.redraw()


# =============================================================================
# 3. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    Explorer()
    plt.show()
