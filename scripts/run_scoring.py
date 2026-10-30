"""SCORING: every journal-model scores the same held-out abstracts ("would we publish this?", 0-100).

Used for:
  A. a 3x3 attribution matrix: assign each held-out paper to the journal-model that rates it highest
     (scores centered within each model to remove leniency differences); plain vs. contrastive.
  B. calibration ("is this a J paper?"): when J's model gives a high score, is the paper really J's
     rather than one of the other two journals'? (share own by score band; AUC own vs. other)

    .venv/bin/python -m scripts.run_scoring --modes plain,contrast --n 40
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from jm.data import JOURNALS, load_works
from jm.journal_model import JournalModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scoring"
BANDS = [-1, 20, 40, 60, 80, 100]
BAND_LABELS = ["0-20", "21-40", "41-60", "61-80", "81-100"]


def held_out(n: int, seed: int = 0) -> pd.DataFrame:
    """n recent (2024-26) test-split core papers per journal (fewer if the journal has fewer)."""
    df = load_works(core_only=True, split="test", years=(2024, 2026))
    rng = random.Random(f"scoring-{seed}")
    rows = []
    for j in JOURNALS:
        ids = sorted(df[df.journal == j].openalex_id)
        rows += [{"id": x, "source": j} for x in rng.sample(ids, min(n, len(ids)))]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="plain,contrast")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    probes = held_out(args.n)
    print(probes.groupby("source").size())
    long = []
    for mode in args.modes.split(","):
        for j in JOURNALS:
            s = JournalModel(j, mode=mode).score(list(probes.id), tag=f"scoring/{mode}")
            long += [{"mode": mode, "model": j, "id": k, "score": v["score"], "why": v["why"]} for k, v in s.items()]
            print(mode, j, len(s))
    df = pd.DataFrame(long).merge(probes, on="id")
    df.to_csv(OUT / "scores.csv", index=False)

    summary = {}
    for mode, g in df.groupby("mode"):
        g = g.copy()
        g["z"] = g.groupby("model").score.transform(lambda s: s - s.mean())
        wide = g.pivot_table(index=["id", "source"], columns="model", values="z").dropna().reset_index()
        wide["assigned"] = wide[list(JOURNALS)].idxmax(axis=1)
        conf = pd.crosstab(wide.source, wide.assigned, normalize="index").reindex(index=JOURNALS, columns=JOURNALS)
        bands = {}
        for j in JOURNALS:
            gj = g[g.model == j].copy()
            gj["band"] = pd.cut(gj.score, BANDS, labels=BAND_LABELS)
            bands[j] = gj.groupby("band", observed=False).agg(
                n=("id", "size"), share_own=("source", lambda s: float((s == j).mean()) if len(s) else None)
            ).to_dict(orient="index")
        summary[mode] = {
            "attribution_accuracy": float((wide.source == wide.assigned).mean()),
            "attribution_matrix": conf.round(3).to_dict(orient="index"),
            "auc_own_vs_other": {j: float(roc_auc_score(g[g.model == j].source == j, g[g.model == j].score))
                                 for j in JOURNALS},
            "mean_score": g.groupby(["model", "source"]).score.mean().round(1).unstack().to_dict(orient="index"),
            "calibration_bands": bands,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({m: {k: v for k, v in r.items() if k != "calibration_bands"} for m, r in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
