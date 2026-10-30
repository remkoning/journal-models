"""Vertical vs. horizontal: do the journal-models agree on which papers are good?

Following Alidaee, Khanna & Koning (2026), who compare firm-specific hiring models on the same
applications, every journal-model scores every held-out paper (the paper x journal matrix of exhibit B).
A second scoring pass with a different batch order gives a noise floor. Two-way random-effects ANOVA
with replication then splits the score variance into:

    paper           what all three journals agree on (vertical quality)
    paper x journal journal-specific fit (horizontal taste)
    noise           scoring noise (replicate disagreement)

after removing each model's mean (leniency, AKK's "sorting without complementarities").
Because the papers come from the three journals, sorting alone makes the models disagree; the
`within_venue` version also removes each model's mean within each publishing journal, asking whether
the models agree on which of, say, Org Sci's own papers are better.

    .venv/bin/python -m scripts.vertical_horizontal
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jm.data import JOURNALS
from jm.journal_model import JournalModel
from scripts.run_scoring import held_out

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scoring"


def components(S: np.ndarray, venue: np.ndarray | None = None) -> dict:
    """S: papers x journals x replicates. Variance components after removing journal-replicate means
    (and, given each paper's publishing venue, journal-replicate means within venue)."""
    S = S - S.mean(0, keepdims=True)                      # leniency, per model and replicate
    if venue is not None:
        for v in np.unique(venue):
            S[venue == v] -= S[venue == v].mean(0, keepdims=True)
    n, J, r = S.shape
    cell, pm, jm, gm = S.mean(2), S.mean((1, 2)), S.mean((0, 2)), S.mean()
    ms_p = J * r * ((pm - gm) ** 2).sum() / (n - 1)
    ms_pj = r * ((cell - pm[:, None] - jm[None] + gm) ** 2).sum() / ((n - 1) * (J - 1))
    ms_e = ((S - cell[..., None]) ** 2).sum() / (n * J * (r - 1))
    v_p, v_pj, v_e = max(0, (ms_p - ms_pj) / (J * r)), max(0, (ms_pj - ms_e) / r), ms_e
    retest = [float(np.corrcoef(S[:, j, 0], S[:, j, 1])[0, 1]) for j in range(J)]
    cross = np.corrcoef(cell.T)
    return {"share_vertical": v_p / (v_p + v_pj), "share_horizontal": v_pj / (v_p + v_pj),
            "noise_share_of_total": v_e / (v_p + v_pj + v_e),
            "retest_r": dict(zip(JOURNALS, retest)),
            "cross_model_r": {f"{JOURNALS[a]}-{JOURNALS[b]}": float(cross[a, b])
                              for a in range(J) for b in range(a + 1, J)}}


def main():
    first = pd.read_csv(OUT / "scores.csv")
    probes = held_out(40)
    res = {}
    for mode in ("plain", "contrast", "rulebook"):
        a = first[first["mode"] == mode].pivot_table(index="id", columns="model", values="score")
        b = pd.DataFrame({j: {k: v["score"] for k, v in JournalModel(j, mode=mode).score(
            list(probes.id), seed=1, tag=f"retest/{mode}").items()} for j in JOURNALS})
        ids = a.index.intersection(b.dropna().index)
        S = np.stack([a.loc[ids, JOURNALS].values, b.loc[ids, JOURNALS].values], axis=2).astype(float)
        venue = first.drop_duplicates("id").set_index("id").loc[ids, "source"].values
        res[mode] = {"n_papers": len(ids), **components(S.copy()), "within_venue": components(S.copy(), venue)}
        print(mode, json.dumps(res[mode], indent=1))
    (OUT / "vertical_horizontal.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
