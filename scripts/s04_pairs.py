"""Step 4: EVALUATE pairs (IDs only, committed) -> evals/pairs/.

cross_{split}.jsonl : topic-matched cross-journal pairs (in_core only), greedy one-to-one matching by cosine
                      similarity, |year diff| <= 1, sim >= threshold. Candidate edges must share the OpenAlex
                      subfield OR have cosine >= HI_SIM (topic block). Plain thresholding without the block could not
                      reach 70% topic/subfield agreement at usable N (diagnostic table kept in pair_stats.json).
cite_{split}.jsonl  : within-journal-year 'promise' pairs, 2016-2020, top vs bottom quartile of 5-year citations.
cross_fullms_{split}.jsonl : robustness, as cross but MgmtSci side uses ALL MgmtSci departments.
Pairs never cross splits. Order within a pair is randomized (seeded).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm.data import JOURNALS, LOCAL, load_embeddings  # noqa: E402

OUT = ROOT / "evals" / "pairs"
SPLITS = ("train", "val", "test")
JPAIRS = [("OrgSci", "SMJ"), ("OrgSci", "MgmtSci"), ("SMJ", "MgmtSci")]
TARGET_SHARE = 0.70
SIM_FLOOR = 0.75          # never accept pairs below this cosine (0.70-0.75 pairs looked unrelated on inspection)
HI_SIM = 0.85             # above this cosine, pairs are topical regardless of the (noisy) OpenAlex subfield label
SEED = 20261030


def _rng(*keys) -> np.random.Generator:
    h = int(hashlib.sha256("|".join(map(str, (SEED,) + keys)).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


def greedy_match(A: pd.DataFrame, B: pd.DataFrame, XA: np.ndarray, XB: np.ndarray, max_dy: int = 1,
                 block: bool = True) -> list[tuple]:
    """Greedy one-to-one matching on cosine similarity among |year diff| <= max_dy (and, if block, same subfield or
    cosine >= HI_SIM). Returns (ia, ib, sim)."""
    if len(A) == 0 or len(B) == 0:
        return []
    S = XA @ XB.T
    dy = np.abs(A["year"].to_numpy()[:, None] - B["year"].to_numpy()[None, :])
    ok = dy <= max_dy
    if block:
        sa, sb = A["subfield"].to_numpy(object), B["subfield"].to_numpy(object)
        ok &= (sa[:, None] == sb[None, :]) | (S >= HI_SIM)
    S = np.where(ok, S, -np.inf)
    ia, ib = np.where(np.isfinite(S))
    order = np.argsort(-S[ia, ib], kind="stable")
    used_a, used_b, out = set(), set(), []
    for k in order:
        a, b = ia[k], ib[k]
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        out.append((a, b, float(S[a, b])))
    return out


def build_cross(df: pd.DataFrame, X: np.ndarray, row: dict, ms_full: bool = False, block: bool = True) -> pd.DataFrame:
    recs = []
    jpairs = JPAIRS if not ms_full else [p for p in JPAIRS if "MgmtSci" in p]
    for split in SPLITS:
        for J, K in jpairs:
            def side(j):
                m = (df.journal == j) & (df.split == split)
                if not (ms_full and j == "MgmtSci"):
                    m &= df.in_core
                return df[m].reset_index(drop=True)
            A, B = side(J), side(K)
            XA, XB = X[[row[i] for i in A.openalex_id]], X[[row[i] for i in B.openalex_id]]
            for a, b, s in greedy_match(A, B, XA, XB, block=block):
                ra, rb = A.iloc[a], B.iloc[b]
                recs.append({"journal_pair": f"{J}-{K}", "split": split, "sim": round(s, 4),
                             "p_id": ra.openalex_id, "q_id": rb.openalex_id, "p_journal": J, "q_journal": K,
                             "p_year": int(ra.year), "q_year": int(rb.year),
                             "same_primary_topic": bool(ra.primary_topic == rb.primary_topic and ra.primary_topic),
                             "same_subfield": bool(ra.subfield == rb.subfield and ra.subfield)})
    m = pd.DataFrame(recs)
    m["same_topic"] = m.same_primary_topic | m.same_subfield
    return m


def choose_threshold(m: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Lowest cosine threshold (>= SIM_FLOOR) at which >= 70% of matched pairs share primary topic or subfield,
    in every journal pair. Greedy matches above t do not depend on t, so filtering post hoc is exact."""
    grid = np.round(np.arange(0.60, 0.951, 0.01), 2)
    rows = []
    for t in grid:
        for jp, g in m.groupby("journal_pair"):
            h = g[g.sim >= t]
            rows.append({"t": t, "journal_pair": jp, "n": len(h), "n_test": int((h.split == "test").sum()),
                         "share_topic_or_subfield": round(float(h.same_topic.mean()), 3) if len(h) else None,
                         "share_primary_topic": round(float(h.same_primary_topic.mean()), 3) if len(h) else None})
    tab = pd.DataFrame(rows)
    ok = tab.groupby("t").share_topic_or_subfield.min()
    feasible = [t for t in grid if ok.get(t) is not None and ok[t] >= TARGET_SHARE and t >= SIM_FLOOR]
    return (min(feasible) if feasible else SIM_FLOOR), tab


def finalize_cross(m: pd.DataFrame, thr: float, prefix: str) -> pd.DataFrame:
    m = m[m.sim >= thr].copy().sort_values(["split", "journal_pair", "sim"], ascending=[True, True, False])
    out = []
    for i, r in enumerate(m.itertuples(index=False)):
        flip = _rng(prefix, r.p_id, r.q_id).random() < 0.5
        a, b = ("q", "p") if flip else ("p", "q")
        g = r._asdict()
        out.append({"pair_id": f"{prefix}-{r.split}-{r.journal_pair}-{hashlib.sha1((r.p_id + r.q_id).encode()).hexdigest()[:8]}",
                    "a_id": g[f"{a}_id"], "b_id": g[f"{b}_id"], "a_journal": g[f"{a}_journal"], "b_journal": g[f"{b}_journal"],
                    "a_year": g[f"{a}_year"], "b_year": g[f"{b}_year"], "sim": r.sim, "same_topic": bool(r.same_topic),
                    "same_primary_topic": bool(r.same_primary_topic), "same_subfield": bool(r.same_subfield),
                    "journal_pair": r.journal_pair, "split": r.split})
    return pd.DataFrame(out)


def build_cite(df: pd.DataFrame, X: np.ndarray, row: dict) -> tuple[pd.DataFrame, list]:
    """Within journal-year (in_core, 2016-2020): quartiles on cites_5yr computed across all splits; pairs formed
    within split. Each high paper is matched to an unused low paper, preferring same subfield, then highest cosine."""
    core = df[df.in_core & df.year.between(2016, 2020) & df.cites_5yr.notna()]
    recs, qstats = [], []
    for (j, y), g in core.groupby(["journal", "year"]):
        q25, q75 = np.quantile(g.cites_5yr, [0.25, 0.75])
        qstats.append({"journal": j, "year": int(y), "n": len(g), "q25": float(q25), "q75": float(q75)})
        for split in SPLITS:
            gs = g[g.split == split]
            hi = gs[gs.cites_5yr >= q75].sample(frac=1, random_state=int(_rng("hi", j, y, split).integers(1e9)))
            lo = gs[gs.cites_5yr <= q25]
            if len(hi) == 0 or len(lo) == 0:
                continue
            lo_used = set()
            Xlo = X[[row[i] for i in lo.openalex_id]]
            for h in hi.itertuples(index=False):
                avail = np.array([i not in lo_used for i in lo.openalex_id])
                if not avail.any():
                    break
                sims = Xlo @ X[row[h.openalex_id]]
                same = (lo.subfield.to_numpy() == h.subfield) & avail
                pool = same if same.any() else avail
                k = int(np.argmax(np.where(pool, sims, -np.inf)))
                l = lo.iloc[k]
                lo_used.add(l.openalex_id)
                flip = _rng("cite", h.openalex_id, l.openalex_id).random() < 0.5
                hi_rec, lo_rec = (h.openalex_id, h.cites_5yr), (l.openalex_id, l.cites_5yr)
                (aid, ac), (bid, bc) = (lo_rec, hi_rec) if flip else (hi_rec, lo_rec)
                recs.append({"pair_id": f"cite-{split}-{j}-{y}-{hashlib.sha1((h.openalex_id + l.openalex_id).encode()).hexdigest()[:8]}",
                             "a_id": aid, "b_id": bid, "journal": j, "year": int(y), "a_cites5": int(ac), "b_cites5": int(bc),
                             "higher": "b" if flip else "a", "same_subfield": bool(same.any()),
                             "sim": round(float(sims[k]), 4), "q25": float(q25), "q75": float(q75), "split": split})
    return pd.DataFrame(recs), qstats


def write_split(d: pd.DataFrame, kind: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with open(OUT / f"{kind}_{split}.jsonl", "w") as f:
            for r in d[d.split == split].to_dict("records"):
                f.write(json.dumps(r) + "\n")


def main():
    df = pd.read_parquet(LOCAL / "works.parquet")
    ids, X = load_embeddings()
    row = {k: i for i, k in enumerate(ids)}
    assert set(df.openalex_id) <= set(row), "embeddings missing for some works; rerun step 3"

    _, tab_unblocked = choose_threshold(build_cross(df, X, row, block=False))   # diagnostic only
    m = build_cross(df, X, row)
    thr, tab = choose_threshold(m)
    cross = finalize_cross(m, thr, "cross")
    write_split(cross, "cross")
    mf = build_cross(df, X, row, ms_full=True)
    cross_full = finalize_cross(mf, thr, "crossfull")
    write_split(cross_full, "cross_fullms")
    cite, qstats = build_cite(df, X, row)
    write_split(cite, "cite")

    def counts(d, by):
        return {f"{k[0]}|{k[1]}" if isinstance(k, tuple) else k: int(v) for k, v in d.groupby(by).size().items()}

    stats = {
        "cross_threshold": thr, "target_share": TARGET_SHARE, "sim_floor": SIM_FLOOR, "hi_sim_override": HI_SIM,
        "cross_rule": "greedy 1:1 by cosine; |dy|<=1; same OpenAlex subfield OR cosine>=HI_SIM; cosine>=threshold",
        "cross_unblocked_threshold_table": tab_unblocked[tab_unblocked.t.isin(
            [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9])].to_dict("records"),
        "cross_threshold_table": tab[tab.t.isin(np.round(np.arange(0.60, 0.951, 0.05), 2).tolist() + [thr])].to_dict("records"),
        "cross_blocked_unthresholded_sim_quantiles_p10_p25_p50_p75_p90": {
            jp: g.sim.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() for jp, g in m.groupby("journal_pair")},
        "cross_counts": counts(cross, ["journal_pair", "split"]),
        "cross_sim": {jp: {"mean": round(g.sim.mean(), 3), "min": round(g.sim.min(), 3), "median": round(g.sim.median(), 3),
                           "share_same_topic_or_subfield": round(g.same_topic.mean(), 3),
                           "share_same_primary_topic": round(g.same_primary_topic.mean(), 3),
                           "share_same_subfield": round(g.same_subfield.mean(), 3),
                           "mean_abs_year_diff": round(float((g.a_year - g.b_year).abs().mean()), 2),
                           "share_a_is_first_journal": round(float((g.a_journal == jp.split('-')[0]).mean()), 3)}
                      for jp, g in cross.groupby("journal_pair")},
        "cross_fullms_counts": counts(cross_full, ["journal_pair", "split"]),
        "cross_fullms_sim": {jp: {"mean": round(g.sim.mean(), 3), "share_same_topic_or_subfield": round(g.same_topic.mean(), 3)}
                             for jp, g in cross_full.groupby("journal_pair")},
        "cite_counts": counts(cite, ["journal", "split"]),
        "cite_share_same_subfield": round(float(cite.same_subfield.mean()), 3),
        "cite_share_higher_a": round(float((cite.higher == "a").mean()), 3),
        "cite_median_ratio_hi_lo": round(float((cite[["a_cites5", "b_cites5"]].max(axis=1) /
                                                cite[["a_cites5", "b_cites5"]].min(axis=1).clip(lower=1)).median()), 2),
        "cite_quartiles": qstats,
    }
    (ROOT / "data" / "manifest" / "pair_stats.json").write_text(json.dumps(stats, indent=1, default=float))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("cross_threshold_table", "cite_quartiles")}, indent=1, default=float))
    print(tab[tab.t.isin([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, thr])].to_string())


if __name__ == "__main__":
    main()
