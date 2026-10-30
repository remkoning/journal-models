"""Figures for the journal-models prototype (PNG, print and grayscale safe).

    .venv/bin/python -m scripts.make_figures

Each figure is drawn only if its inputs exist:
  fig_A_pairs.png        accuracy picking own paper on topic-matched test pairs, by journal pair x condition
  fig_A_attribution.png  3x3: held-out papers assigned to the journal-model that rates them highest
  fig_B_calibration.png  share of papers truly from J, by J-model score band
  fig_C_push.png         where each journal pushes research: blind-classified suggestion categories
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jm.data import JOURNALS, JOURNAL_NAMES, load_pairs

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

# Palette: validated reference instance (dataviz skill). Journals = categorical slots 1-3 (all-pairs pass);
# conditions = ordinal single-hue blue ramp (steps 250 / 400 / 550 / 700).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
JCOLOR = {"OrgSci": "#2a78d6", "SMJ": "#eb6834", "MgmtSci": "#1baf7a"}
JMARK = {"OrgSci": "o", "SMJ": "s", "MgmtSci": "^"}
SHORT = {"OrgSci": "Org Sci", "SMJ": "SMJ", "MgmtSci": "Mgmt Sci"}
CONDS = [("plain", "Name only", "#86b6ef"), ("incontext", "+100 own abstracts", "#3987e5"),
         ("contrast", "+100 own & 100 elsewhere", "#1c5cab"), ("rulebook", "Evolved rulebook", "#0d366b")]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def _pair_rows(sfx: str) -> pd.DataFrame:
    pairs = {p["pair_id"]: p for p in load_pairs("cross", "test")}
    rows = []
    for mode, _, _ in CONDS:
        for j in JOURNALS:
            f = RES / "pilot" / f"probs_test_{mode}_{j}{sfx}.json"
            if not f.exists():
                continue
            for pid, pa in json.loads(f.read_text()).items():
                p = pairs[pid]
                own = pa if p["a_journal"] == j else 1 - pa
                rows.append({"mode": mode, "jp": p["journal_pair"],
                             "correct": 1.0 if own > 0.5 else 0.5 if own == 0.5 else 0.0})   # tie = half credit
    return pd.DataFrame(rows)


def _pair_panel(ax, d, title, labels):
    jps = ["OrgSci-SMJ", "SMJ-MgmtSci", "OrgSci-MgmtSci"]
    conds = [c for c in CONDS if c[0] in set(d["mode"])]
    w = 0.8 / len(conds)
    for k, (mode, _, color) in enumerate(conds):
        for i, jp in enumerate(jps):
            x = d[(d["mode"] == mode) & (d.jp == jp)].correct.values
            if len(x) == 0:
                continue
            lo, hi = boot(x)
            xi = i + (k - (len(conds) - 1) / 2) * w
            ax.bar(xi, x.mean(), w - 0.02, color=color, label=labels.get(mode) if i == 0 else None, zorder=2)
            ax.plot([xi, xi], [lo, hi], color=INK, lw=1, zorder=3)
            ax.text(xi, hi + 0.012, f"{x.mean():.0%}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(jps)), [f"{SHORT[a]}\nvs {SHORT[b]}" for a, b in (jp.split("-") for jp in jps)])
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--", zorder=1)
    ax.text(-0.55, 0.51, "chance", fontsize=8, color=INK2, ha="left")
    ax.set_ylim(0.3, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2))


def fig_pairs():
    # Prefer one-pair-per-call runs (_b1); earlier batch-of-10 runs had in-batch leakage.
    named = _pair_rows("_b1")
    anon = _pair_rows("_anon_b1")
    if named.empty:
        return
    panels = [(named, "Journal named", {"plain": "Name only", "incontext": "+ 100 own abstracts",
                                        "contrast": "+ 100 own & 100 elsewhere"})]
    if not anon.empty:
        panels.append((anon, 'Journal hidden ("Journal X")', {"plain": "Nothing (no name, no abstracts)",
                                                              "incontext": "100 own abstracts only",
                                                              "contrast": "100 own & 100 elsewhere",
                                                              "rulebook": "~20 evolved rules only"}))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (d, title, labels) in zip(axes, panels):
        _pair_panel(ax, d, title, labels)
    axes[0].set_ylabel("Picks its own journal's paper")
    fig.suptitle("Is journal taste learnable? Topic-matched held-out pairs (one pair per call, both orders)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_A_pairs.png", dpi=200)
    plt.close(fig)


def fig_attribution():
    f = RES / "scoring" / "summary.json"
    if not f.exists():
        return
    s = json.loads(f.read_text())
    modes = [m for m in ("plain", "contrast") if m in s]
    fig, axes = plt.subplots(1, len(modes), figsize=(4.0 * len(modes), 3.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, modes):
        M = pd.DataFrame(s[m]["attribution_matrix"]).T.reindex(index=JOURNALS, columns=JOURNALS).values
        ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUES), vmin=0, vmax=1)
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f"{M[i, k]:.0%}", ha="center", va="center", fontsize=10,
                        color="white" if M[i, k] > 0.55 else INK)
        ax.set_xticks(range(3), [SHORT[j] for j in JOURNALS])
        ax.set_yticks(range(3), [SHORT[j] for j in JOURNALS])
        ax.set_xlabel("Claimed by (model rating it highest)")
        ax.grid(False)
        title = "Name only" if m == "plain" else "Contrastive abstracts"
        ax.set_title(f"{title}: {s[m]['attribution_accuracy']:.0%} correct", fontsize=10, loc="left")
    axes[0].set_ylabel("Actually published in")
    fig.suptitle("Which journal would claim this held-out paper? (chance = 33%)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_A_attribution.png", dpi=200)
    plt.close(fig)


def fig_calibration():
    f = RES / "scoring" / "summary.json"
    if not f.exists():
        return
    s = json.loads(f.read_text())
    m = "contrast" if "contrast" in s else next(iter(s))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for j in JOURNALS:
        b = s[m]["calibration_bands"][j]
        xs, ys, ns = [], [], []
        for k, (band, v) in enumerate(b.items()):
            if v["n"] >= 5 and v["share_own"] is not None:
                xs.append(k); ys.append(v["share_own"]); ns.append(v["n"])
        ax.plot(xs, ys, color=JCOLOR[j], lw=2, marker=JMARK[j], ms=8, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(xs[-1] + 0.08, ys[-1], SHORT[j], color=INK, fontsize=9, va="center")
    ax.axhline(1 / 3, color=INK2, lw=0.8, ls="--")
    ax.text(0, 1 / 3 + 0.02, "base rate (1 in 3)", fontsize=8, color=INK2)
    ax.set_xticks(range(4), list(next(iter(s[m]["calibration_bands"].values())).keys())[:4])
    ax.set_xlim(-0.2, 3.7)
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Journal-model's score: \"would we publish this?\" (bands with n ≥ 5)")
    ax.set_ylabel("Share truly from that journal")
    ax.set_title("Calibration: a high score means it really is that journal's paper", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_B_calibration.png", dpi=200)
    plt.close(fig)


def fig_push():
    f = RES / "feedback" / "summary_contrast.json"
    if not f.exists():
        return
    s = json.loads(f.read_text())
    cats = ["mechanism", "theory", "methods", "data", "scope", "positioning", "practice"]
    labels = ["Mechanism", "Theory", "Methods / identification", "Data / measurement", "Scope / context",
              "Positioning", "Practice / implications"]
    M = np.array([[s["category_share"][j][c] for j in JOURNALS] for c in cats])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUES), vmin=0, vmax=0.32,
              aspect="auto")
    for i in range(len(cats)):
        for k in range(3):
            ax.text(k, i, f"{M[i, k]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, k] > 0.2 else INK)
    ax.set_xticks(range(3), [SHORT[j] for j in JOURNALS])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(cats)), labels)
    ax.grid(False)
    n = s["n_reviews"] * 3
    ax.set_title(f"Where each journal pushes the research\n{n} next-step suggestions on the same "
                 f"{s['n_papers']} papers,\nclassified blind to journal", loc="left", fontsize=10, pad=26)
    fig.tight_layout()
    fig.savefig(FIG / "fig_C_push.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_pairs, fig_attribution, fig_calibration, fig_push):
        f()
    print(sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
