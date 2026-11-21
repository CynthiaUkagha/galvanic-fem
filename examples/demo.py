"""Make the README figure.

    python examples/demo.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from galvanicfem import Params, protection_distance, solve

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)
CASES = [(1e-5, "10 µm (dew, road spray)"), (1e-4, "0.1 mm"), (1e-3, "1 mm (wet film)"), (1e-2, "10 mm (immersed)")]
COLORS = ["#d1495b", "#edae49", "#00798c", "#2747a8"]


def main():
    p = Params()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    for (h, label), c in zip(CASES, COLORS):
        s = solve(h)
        ax[0].plot(s.x * 1e3, s.E, color=c, lw=1.8, label=label)
        ax[1].plot(s.x * 1e3, s.i, color=c, lw=1.5)
    for a in ax[:2]:
        a.axvline(0, color="0.5", lw=0.8, ls="--")
        a.set_xlabel("position along surface (mm)   ← zinc | steel →")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].axhline(p.E_st - 0.05, color="0.4", lw=0.8, ls=":")
    ax[0].text(29, p.E_st - 0.045, "protected below this line", ha="right", fontsize=8, color="0.3")
    ax[0].set_ylabel("electrode potential (V vs SCE)")
    ax[0].set_title("1. Surface potential", fontweight="bold", loc="left")
    ax[0].legend(frameon=False, fontsize=8, title="electrolyte film", title_fontsize=8, loc="upper left")
    ax[1].set_ylabel("current density (A/m²)   + zinc dissolving")
    ax[1].set_title("2. Corrosion current", fontweight="bold", loc="left")
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[1].axhline(0, color="0.5", lw=0.6)

    hs = np.logspace(-5, -2.3, 14)
    d = [protection_distance(solve(h)) * 1e3 for h in hs]
    ax[2].plot(hs * 1e3, d, "o-", color="#2747a8", ms=4)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("electrolyte film thickness (mm)")
    ax[2].set_ylabel("steel protected by zinc (mm)")
    ax[2].set_title("3. Protection distance vs film", fontweight="bold", loc="left")
    ax[2].text(hs[-1] * 1e3, 30.6, "whole modelled steel strip protected", ha="right", fontsize=8, color="0.3")
    for sp in ("top", "right"):
        ax[2].spines[sp].set_visible(False)
    fig.savefig(OUT / "overview.png", dpi=130)
    for h, dist in zip(hs, d):
        print(f"{h * 1e3:7.4f} mm film -> {dist:5.1f} mm protected")


if __name__ == "__main__":
    main()
