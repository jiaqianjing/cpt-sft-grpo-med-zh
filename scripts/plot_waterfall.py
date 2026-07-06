"""Render the CPT→SFT→GRPO gain-attribution waterfall to assets/gain_waterfall.png.

Palette (validated CVD-safe, dataviz skill): blue #2a78d6 = gain, orange #eb6834 = loss,
neutral gray = anchors (base/final). Every bar is value-labelled with a sign, so polarity
is never color-alone. Numbers are the published Qwen3-1.7B results (REPORT.md §4).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- palette ---
SURF, INK, INK2, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, GAIN, LOSS, ANCHOR = "#e1e0d9", "#2a78d6", "#eb6834", "#b8b7b0"

# --- results (REPORT.md §4) ---
GEN = {"R0 base": 0.589, "R1 SFT": 0.544, "R3 CPT+SFT": 0.537, "R4 +GRPO": 0.552}
STAGES = [("+SFT", 0.589, 0.544), ("+CPT", 0.544, 0.537), ("+GRPO", 0.537, 0.552)]
PPL = {"base": 7.04, "CPT": 5.79}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": MUT, "text.color": INK, "axes.labelcolor": INK2})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 5.0),
                               gridspec_kw={"width_ratios": [2.5, 1]})
fig.patch.set_facecolor(SURF)

# ================= Panel A: generative-accuracy waterfall =================
axA.set_facecolor(SURF)
labels = ["R0\nbase", "+SFT", "+CPT", "+GRPO", "R4\nfinal"]
x = range(5)
floor = 0.50
# anchors
axA.bar(0, GEN["R0 base"] - floor, bottom=floor, width=0.6, color=ANCHOR, zorder=3)
axA.bar(4, GEN["R4 +GRPO"] - floor, bottom=floor, width=0.6, color=ANCHOR, zorder=3)
# deltas
for i, (name, prev, cur) in enumerate(STAGES, start=1):
    lo, hi = min(prev, cur), max(prev, cur)
    col = GAIN if cur >= prev else LOSS
    axA.bar(i, hi - lo, bottom=lo, width=0.6, color=col, zorder=3)
    d = cur - prev
    axA.text(i, hi + 0.002, f"{d:+.3f}", ha="center", va="bottom",
             color=(GAIN if d >= 0 else LOSS), fontsize=10, fontweight="bold")
    # connector
    axA.plot([i - 1 + 0.3, i - 0.3], [prev, prev], color=MUT, lw=1, ls=(0, (3, 3)), zorder=2)
axA.plot([3 + 0.3, 4 - 0.3], [0.552, 0.552], color=MUT, lw=1, ls=(0, (3, 3)), zorder=2)
# anchor value labels
axA.text(0, GEN["R0 base"] + 0.002, f'{GEN["R0 base"]:.3f}', ha="center", va="bottom", color=INK, fontweight="bold")
axA.text(4, GEN["R4 +GRPO"] + 0.002, f'{GEN["R4 +GRPO"]:.3f}', ha="center", va="bottom", color=INK, fontweight="bold")
axA.set_xticks(list(x)); axA.set_xticklabels(labels, color=INK2)
axA.set_ylim(floor, 0.605); axA.set_ylabel("Generative MCQ accuracy")
axA.set_title("Per-stage gain attribution  ·  generative accuracy", color=INK, fontsize=12, pad=10, loc="left")
axA.grid(axis="y", color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)
for s in ("top", "right"): axA.spines[s].set_visible(False)
axA.spines["left"].set_color(MUT); axA.spines["bottom"].set_color(MUT)
axA.legend(handles=[Patch(color=ANCHOR, label="level (base / final)"),
                    Patch(color=GAIN, label="gain (↑)"),
                    Patch(color=LOSS, label="loss (↓)")],
           loc="upper right", frameon=False, fontsize=9)
axA.text(0.0, -0.16, "y-axis starts at 0.50 to make per-stage deltas legible.",
         transform=axA.transAxes, color=MUT, fontsize=8.5)

# ================= Panel B: perplexity (CPT's clear win) =================
axB.set_facecolor(SURF)
axB.bar(0, PPL["base"], width=0.55, color=ANCHOR, zorder=3)
axB.bar(1, PPL["CPT"], width=0.55, color=GAIN, zorder=3)
for i, (k, v) in enumerate(PPL.items()):
    axB.text(i, v + 0.06, f"{v:.2f}", ha="center", va="bottom", color=INK, fontweight="bold")
axB.annotate("", xy=(0.5, PPL["CPT"]), xytext=(0.5, PPL["base"]),
             arrowprops=dict(arrowstyle="->", color=GAIN, lw=2))
axB.text(0.6, (PPL["base"] + PPL["CPT"]) / 2, "-1.25\n(-18%)", color=GAIN, fontsize=10, fontweight="bold", va="center", ha="left")
axB.set_xticks([0, 1]); axB.set_xticklabels(["R0\nbase", "R2\nCPT"], color=INK2)
axB.set_ylim(0, 7.8); axB.set_ylabel("Held-out perplexity  (lower = better)")
axB.set_title("CPT's clear win  ·  perplexity", color=INK, fontsize=12, pad=10, loc="left")
axB.grid(axis="y", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)
for s in ("top", "right"): axB.spines[s].set_visible(False)
axB.spines["left"].set_color(MUT); axB.spines["bottom"].set_color(MUT)

fig.suptitle("CPT → SFT → GRPO gain attribution  ·  Qwen3-1.7B  ·  Chinese medical",
             fontsize=13.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.text(0.02, 0.015,
         "GRPO trained correctly (reward 0.40→0.625) but transferred little to test; SFT/GRPO downstream "
         "gains were capped by a weak distillation teacher (Gemini flash-lite). CPT gives the one clear gain. See REPORT.md.",
         color=MUT, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
import os
os.makedirs("assets", exist_ok=True)
fig.savefig("assets/gain_waterfall.png", dpi=150, facecolor=SURF, bbox_inches="tight")
print("wrote assets/gain_waterfall.png")
