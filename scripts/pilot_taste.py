"""Pilot: is 100 abstracts in context enough to learn a journal's taste?

On topic-matched cross-journal pairs, each journal-model says which paper its journal published.
Conditions: plain / incontext / contrast, journal named or anonymized ("Journal X").
Use --batch 1 (default): batching several pairs that all contain the target journal's paper lets the
model infer the target journal from the batch itself (in-batch leakage). Early runs used batch 10.

    .venv/bin/python -m scripts.pilot_taste [--split val] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from jm.data import JOURNALS, load_pairs
from jm.journal_model import JournalModel
from jm.llm import spend_report

OUT = Path(__file__).resolve().parents[1] / "results" / "pilot"


def accuracy(pairs, probs, journal):
    """1 if the model puts >50% on its own journal's paper, 0.5 on exact ties."""
    acc = []
    for p in pairs:
        if p["pair_id"] not in probs:
            continue
        pa = probs[p["pair_id"]]
        p_own = pa if p["a_journal"] == journal else 1 - pa
        acc.append(1.0 if p_own > 0.5 else 0.5 if p_own == 0.5 else 0.0)
    return np.array(acc)


def boot_ci(x, n=2000, seed=0):
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--modes", default="plain,incontext,contrast")
    ap.add_argument("--anonymous", action="store_true")
    ap.add_argument("--batch", type=int, default=1, help="pairs per call; 1 avoids in-batch leakage")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pairs = load_pairs("cross", args.split)
    if args.limit:
        pairs = random.Random(0).sample(pairs, args.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for mode in args.modes.split(","):
        for j in JOURNALS:
            mine = [p for p in pairs if j in (p["a_journal"], p["b_journal"])]
            if not mine:
                continue
            jm_ = JournalModel(j, mode=mode, anonymous=args.anonymous)
            probs = jm_.which_is_ours(mine, batch=args.batch, tag=f"pilot/{args.split}")
            acc = accuracy(mine, probs, j)
            lo, hi = boot_ci(acc)
            rows.append({"mode": mode, "journal": j, "n_pairs": len(acc), "acc": float(acc.mean()), "ci": [lo, hi]})
            print(f"{mode:9s}{'(anon)' if args.anonymous else ''} {j:7s} n={len(acc):3d} acc={acc.mean():.3f} [{lo:.2f},{hi:.2f}]  ({time.time() - t0:.0f}s)")
            sfx = ("_anon" if args.anonymous else "") + (f"_b{args.batch}" if args.batch != 10 else "")
            (OUT / f"probs_{args.split}_{mode}_{j}{sfx}.json").write_text(json.dumps(probs))
    (OUT / f"pilot_{args.split}{'_anon' if args.anonymous else ''}{f'_b{args.batch}' if args.batch != 10 else ''}.json").write_text(json.dumps(rows, indent=2))
    print(spend_report())


if __name__ == "__main__":
    main()
