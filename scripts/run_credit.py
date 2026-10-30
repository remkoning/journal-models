"""CREDIT: how much does each of a journal-model's 100 taste abstracts shape its judgments?

Random-subset ("datamodel"/Banzhaf) estimate of each abstract's influence on the model:
  1. Draw m random subsets of k abstracts from the journal's 100-abstract taste set.
  2. For each subset, the journal-model scores a fixed probe set of held-out abstracts (0-100 "would we
     publish this?"). It sees that subset of its own abstracts plus a FIXED contrast set of k abstracts
     from the other two journals, so only the own-journal subset varies across runs.
  3. For abstract i and probe p: beta[i, p] = mean score of p when i is in the subset minus when it is not.
     value[i]     = mean over the journal's own probes of beta minus mean over other probes
                    (does including i sharpen the model's taste for its own journal's work?)
     influence[i] = RMS of beta over probes, estimated from independent split halves
                    (how much does including i move the model's judgments at all?)
  4. Split-half reliability (mean over random half-splits), then correlation with citations.

Designs:
  contrast  (pilot) the subset plus a fixed contrast set; probe order reshuffled each run
  own       (v2) the subset only, so every bit of the model's taste comes from the abstracts that vary;
            run with --anonymous ("Journal X"), --k 10 so each abstract carries more weight, 48 probes,
            and one fixed probe order so runs differ only in the subset

    .venv/bin/python -m scripts.run_credit --journal OrgSci --m 300 --k 20
    .venv/bin/python -m scripts.run_credit --journal OrgSci --design own --anonymous --k 10 --n-probes 48 --m 600
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from jm.data import JOURNALS, emb, load_works
from jm.journal_model import JournalModel, taste_ids
from jm.llm import call, map_parallel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "credit"

SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"item": {"type": "integer"}, "score": {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": ["item", "score"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}


def probe_set(journal: str, n_own: int = 12, n_other: int = 12, seed: int = 0) -> list[dict]:
    """Recent (2024-26) held-out core abstracts: n_own from the journal, n_other split across the other two."""
    df = load_works(core_only=True, split="test", years=(2024, 2026))
    rng = random.Random(f"probe-{journal}-{seed}")
    own = sorted(df[df.journal == journal].openalex_id)
    others = [j for j in JOURNALS if j != journal]
    out = [{"id": x, "own": True} for x in rng.sample(own, n_own)]
    for j in others:
        pool = sorted(df[df.journal == j].openalex_id)
        out += [{"id": x, "own": False} for x in rng.sample(pool, n_other // len(others))]
    return out


def run_subset(journal, subset, probes, s_idx, model, effort, anonymous=False, design="contrast"):
    jmod = JournalModel(journal, mode="contrast" if design == "contrast" else "incontext", ids=subset,
                        model=model, effort=effort, anonymous=anonymous)
    order = list(range(len(probes)))
    random.Random(s_idx if design == "contrast" else 0).shuffle(order)
    ids = [probes[i]["id"] for i in order]
    r = call(jmod.score_prompt(ids).replace(", and in `why` a short phrase (at most 15 words) naming what drives "
                                             "your judgment", ""),
             system=jmod.system, model=model, effort=effort, schema=SCHEMA, tag=f"credit/{journal}")
    got = {a["item"]: a["score"] for a in (r["json"] or {}).get("answers", [])}
    y = np.full(len(probes), np.nan)
    for k, i in enumerate(order):
        if k + 1 in got:
            y[i] = got[k + 1]
    return y


def betas(X, Y):
    """beta[i, p] = mean(Y[:, p] | X[:, i] = 1) - mean(Y[:, p] | X[:, i] = 0), ignoring NaNs."""
    B = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(X.shape[1]):
        inn, out = X[:, i] == 1, X[:, i] == 0
        B[i] = np.nanmean(Y[inn], axis=0) - np.nanmean(Y[out], axis=0)
    return B


def split_half(X, Y, own, splits=200, labels=None):
    """Mean split-half r of value over random half-splits of the runs (one fixed split is too noisy).
    With labels, value is first averaged within each group: credit to a line of work, not one paper."""
    val = lambda b: b[:, own].mean(1) - b[:, ~own].mean(1)
    agg = (lambda v: v) if labels is None else (lambda v: np.array([v[labels == g].mean() for g in np.unique(labels)]))
    rs = []
    for s in range(splits):
        h = np.random.default_rng(s).permutation(len(X)) < len(X) // 2
        va, vb = agg(val(betas(X[h], Y[h]))), agg(val(betas(X[~h], Y[~h])))
        ok = np.isfinite(va) & np.isfinite(vb)       # small m: an abstract can be missing from a half
        rs.append(np.corrcoef(va[ok], vb[ok])[0, 1])
    r = float(np.mean(rs))
    return r, 2 * r / (1 + r)                                           # Spearman-Brown to all runs


def analyze(journal, X, Y, probes, ids):
    own = np.array([p["own"] for p in probes])
    half = np.arange(len(X)) % 2 == 0
    BA, BB, B = betas(X[half], Y[half]), betas(X[~half], Y[~half]), betas(X, Y)
    val = lambda b: b[:, own].mean(1) - b[:, ~own].mean(1)
    v = val(B)
    infl_sq = (BA * BB).mean(1)                       # unbiased for mean squared signal
    influence = np.sqrt(np.clip(infl_sq, 0, None))
    r_half, rel = split_half(X, Y, own)
    by_m = {int(m): split_half(X[:m], Y[:m], own, splits=100)[1] for m in (150, 300) if m < len(X)}
    lines = KMeans(10, n_init=10, random_state=0).fit_predict(emb(ids))    # 10 lines of work (bge clusters)
    rel_lines = split_half(X, Y, own, splits=100, labels=lines)[1]
    w = load_works().set_index("openalex_id").loc[ids]
    pct = w.groupby("year")["cites_total"].rank(pct=True).values  # within-year citation percentile
    logc = np.log1p(w["cites_total"].values)
    corr = lambda a, b: float(spearmanr(a, b, nan_policy="omit").statistic)
    res = {
        "journal": journal, "m_subsets": int(len(X)), "k": int(X[0].sum()), "n_probes": len(probes),
        "missing_scores_share": float(np.isnan(Y).mean()),
        "split_half_r_value": r_half, "reliability_value": rel, "reliability_at_m": by_m,
        "reliability_value_10_lines_of_work": rel_lines,
        "mean_own_minus_other_score": float(np.nanmean(Y[:, own]) - np.nanmean(Y[:, ~own])),
        "spearman_value_vs_cite_pct": corr(v, pct), "spearman_value_vs_logcites": corr(v, logc),
        "spearman_influence_vs_cite_pct": corr(influence, pct),
        "spearman_influence_vs_logcites": corr(influence, logc),
        "per_abstract": [{"id": i, "title": w.loc[i, "title"], "year": int(w.loc[i, "year"]),
                          "cites_total": int(w.loc[i, "cites_total"]), "cite_pct": float(p),
                          "value": float(a), "influence": float(b), "line_of_work": int(c)}
                         for i, p, a, b, c in zip(ids, pct, v, influence, lines)],
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True, choices=JOURNALS)
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--anonymous", action="store_true")
    ap.add_argument("--design", default="contrast", choices=["contrast", "own"])
    ap.add_argument("--n-probes", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = taste_ids(args.journal)
    probes = probe_set(args.journal, args.n_probes // 2, args.n_probes // 2)
    rng = random.Random(f"subsets-{args.journal}")
    subsets = [sorted(rng.sample(range(len(ids)), args.k)) for _ in range(args.m)]
    X = np.zeros((args.m, len(ids)), dtype=int)
    for s, sub in enumerate(subsets):
        X[s, sub] = 1
    rows = map_parallel(lambda s: run_subset(args.journal, [ids[i] for i in subsets[s]], probes, s, args.model,
                                             args.effort, args.anonymous, args.design),
                        range(args.m), workers=args.workers)
    ok = [s for s, r in enumerate(rows) if not isinstance(r, Exception)]
    print(f"{args.journal}: {len(ok)}/{args.m} subsets succeeded")
    Y = np.vstack([rows[s] for s in ok])
    X = X[ok]
    sfx = ("_anon" if args.anonymous else "") + ("" if args.design == "contrast" else f"_own_k{args.k}_p{args.n_probes}")
    np.savez(OUT / f"{args.journal}{sfx}_runs.npz", X=X, Y=Y, ids=np.array(ids), probes=json.dumps(probes))
    res = analyze(args.journal, X, Y, probes, ids)
    (OUT / f"{args.journal}{sfx}_credit.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_abstract"}, indent=2))


if __name__ == "__main__":
    main()
