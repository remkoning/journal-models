"""Step 5: DEVELOP seeds and their realized futures.

Seeds: 30 per journal, 2016-2018, in_core, split in {val, test}, >= 5 citing works published >= seed year + 2 (with
abstracts), stratified across OpenAlex primary topics (topics drawn proportional to size without replacement, one paper
per topic per round).
Realized future per seed:
  (i)  citing_top: up to 10 most-cited citing works published >= seed year + 2, any venue, with abstracts (OpenAlex).
  (ii) semantic_neighbor: 20 nearest 2021-2026 works in our 3-journal corpus (cosine, bge) that do NOT cite the seed;
       citing works met along the way are kept as kind='semantic_citer' (cites_seed=True).
Placebo: another seed with the same primary topic (else same subfield, else any), nearest by embedding.

Outputs: develop/seeds.jsonl, develop/future_manifest.jsonl (IDs only; committed),
         data/local/develop_future.parquet (+ develop_future_embeddings.npy / _ids.json) with text.
Flags: --from-manifest  rebuild local text from the committed manifests (fetches works by ID; ~10 credits).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm import clean as C  # noqa: E402
from jm import openalex as oa  # noqa: E402
from jm.data import JOURNALS, LOCAL, load_embeddings  # noqa: E402

OUT = ROOT / "develop"
N_PER_JOURNAL = 30
N_CITING = 10
N_SEMANTIC = 20
MIN_FUTURE_CITING = 5
SEED = 20261030
CITING_SELECT = "id,doi,title,publication_year,type,cited_by_count,abstract_inverted_index,primary_location"
VENUE_TO_J = {"S206124708": "OrgSci", "S102949365": "SMJ", "S33323087": "MgmtSci"}


def stratified_order(g: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """Order candidates: rounds over topics; within a round topics are drawn PPS (by remaining size) w/o replacement."""
    pools = {t: list(rng.permutation(sub.openalex_id.to_numpy())) for t, sub in g.groupby("primary_topic")}
    order = []
    while any(pools.values()):
        topics = [t for t, v in pools.items() if v]
        w = np.array([len(pools[t]) for t in topics], float)
        for t in rng.choice(topics, size=len(topics), replace=False, p=w / w.sum()):
            order.append(pools[t].pop())
    return order


def citing_works(seed_id: str, year: int) -> tuple[int, list[dict]]:
    d = oa.get("works", {"filter": f"cites:{seed_id},publication_year:>{year + 1},has_abstract:true,"
                                   f"type:article|review|book-chapter",
                         "sort": "cited_by_count:desc", "per-page": N_CITING, "select": CITING_SELECT})
    return int(d["meta"]["count"]), d["results"]


def future_row(w: dict) -> dict:
    src = ((w.get("primary_location") or {}).get("source") or {})
    sid = oa.short_id(src["id"]) if src.get("id") else None
    j = VENUE_TO_J.get(sid, "other")
    raw = oa.abstract_from_inv(w.get("abstract_inverted_index"))
    clean, _ = C.clean_abstract(raw, j)
    return {"future_id": oa.short_id(w["id"]), "year": w.get("publication_year"), "venue": src.get("display_name"),
            "venue_id": sid, "type": w.get("type"), "cited_by_count": w.get("cited_by_count"),
            "title": w.get("title"), "abstract_raw": raw, "abstract_clean": clean,
            "abstract_norm": C.normalize_length(clean, 180)}


def corpus_future_row(r) -> dict:
    return {"future_id": r.openalex_id, "year": int(r.year), "venue": r.journal, "venue_id": None, "type": r.type,
            "cited_by_count": int(r.cites_total), "title": r.title, "abstract_raw": r.abstract_raw,
            "abstract_clean": r.abstract_clean, "abstract_norm": r.abstract_norm}


def build(df: pd.DataFrame, X: np.ndarray, row: dict):
    rng = np.random.default_rng(SEED)
    cand = df[df.in_core & df.year.between(2016, 2018) & df.split.isin(["val", "test"]) & (df.cites_total >= 5)]
    future_pool = df[df.year.between(2021, 2026)].reset_index(drop=True)
    Xf = X[[row[i] for i in future_pool.openalex_id]]
    cites = future_pool.referenced_works.map(set).to_numpy()
    seeds, frows = [], []
    for j in JOURNALS:
        g = cand[cand.journal == j]
        order = stratified_order(g, rng)
        print(j, "candidates", len(g), "topics", g.primary_topic.nunique(), flush=True)
        taken = 0
        for rank, sid in enumerate(order):
            if taken >= N_PER_JOURNAL:
                break
            s = g[g.openalex_id == sid].iloc[0]
            n_fut, works = citing_works(sid, int(s.year))
            if n_fut < MIN_FUTURE_CITING:
                continue
            taken += 1
            for k, w in enumerate(works):
                frows.append({"seed_id": sid, "kind": "citing_top", "rank": k + 1, "cites_seed": True, **future_row(w)})
            sims = Xf @ X[row[sid]]
            n_sem, n_citers = 0, 0
            for k in np.argsort(-sims):
                fr = future_pool.iloc[k]
                c = sid in cites[k]
                frows.append({"seed_id": sid, "kind": "semantic_citer" if c else "semantic_neighbor",
                              "rank": int(n_sem + n_citers + 1), "cites_seed": bool(c), "sim_to_seed": float(sims[k]),
                              **corpus_future_row(fr)})
                n_citers += c
                n_sem += (not c)
                if n_sem >= N_SEMANTIC:
                    break
            seeds.append({"seed_id": sid, "journal": j, "year": int(s.year), "split": s.split,
                          "primary_topic": s.primary_topic, "subfield": s.subfield, "ms_department": s.ms_department,
                          "cites_total": int(s.cites_total), "cites_5yr": None if pd.isna(s.cites_5yr) else int(s.cites_5yr),
                          "n_citing_future_total": n_fut, "n_citing_top": len(works),
                          "n_semantic_citers_before_20": int(n_citers), "strat_order": rank})
        print(j, "seeds", taken, "OpenAlex", oa.usage_summary(), flush=True)
    return pd.DataFrame(seeds), pd.DataFrame(frows)


def add_placebos(seeds: pd.DataFrame, X: np.ndarray, row: dict) -> pd.DataFrame:
    E = X[[row[i] for i in seeds.seed_id]]
    S = E @ E.T
    out = []
    for i, s in seeds.reset_index(drop=True).iterrows():
        others = np.arange(len(seeds)) != i
        for level, mask in (("primary_topic", seeds.primary_topic.to_numpy() == s.primary_topic),
                            ("subfield", seeds.subfield.to_numpy() == s.subfield), ("embedding", np.ones(len(seeds), bool))):
            m = mask & others
            if m.any():
                k = int(np.argmax(np.where(m, S[i], -np.inf)))
                out.append({"placebo_seed_id": seeds.iloc[k].seed_id, "placebo_match": level,
                            "placebo_sim": round(float(S[i, k]), 4)})
                break
    return pd.concat([seeds.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def _nonan(r: dict) -> dict:
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}


def write_outputs(seeds: pd.DataFrame | None, fut: pd.DataFrame, X: np.ndarray, row: dict, write_manifests: bool = True):
    from jm import embed as E
    OUT.mkdir(parents=True, exist_ok=True)
    if write_manifests:
        with open(OUT / "seeds.jsonl", "w") as f:
            for r in seeds.to_dict("records"):
                f.write(json.dumps(_nonan(r)) + "\n")
    fut["in_corpus"] = fut.future_id.isin(row)
    # cosine to seed for citing works too (embed out-of-corpus ones locally)
    ext = fut.loc[~fut.in_corpus].drop_duplicates("future_id")
    Xe = E.embed(ext.abstract_norm.tolist()) if len(ext) else np.zeros((0, X.shape[1]), np.float32)
    np.save(LOCAL / "develop_future_embeddings.npy", Xe)
    (LOCAL / "develop_future_embedding_ids.json").write_text(json.dumps(ext.future_id.tolist()))
    erow = {k: i for i, k in enumerate(ext.future_id)}

    def vec(fid):
        return X[row[fid]] if fid in row else Xe[erow[fid]]
    fut["sim_to_seed"] = [float(vec(r.future_id) @ X[row[r.seed_id]]) for r in fut.itertuples()]
    fut.to_parquet(LOCAL / "develop_future.parquet", index=False)
    if not write_manifests:
        return
    man = fut.drop(columns=["title", "abstract_raw", "abstract_clean", "abstract_norm"])
    with open(OUT / "future_manifest.jsonl", "w") as f:
        for r in man.to_dict("records"):
            f.write(json.dumps(_nonan(r)) + "\n")


def from_manifest(df: pd.DataFrame, X: np.ndarray, row: dict):
    """Rebuild data/local develop text from committed manifests (fetch out-of-corpus works by ID, 50 per call)."""
    man = pd.DataFrame([json.loads(line) for line in open(OUT / "future_manifest.jsonl") if line.strip()])
    need = sorted(set(man.future_id) - set(df.openalex_id))
    got = {}
    for i in range(0, len(need), 50):
        d = oa.get("works", {"filter": "ids.openalex:" + "|".join(need[i:i + 50]), "per-page": 50, "select": CITING_SELECT})
        got.update({oa.short_id(w["id"]): future_row(w) for w in d["results"]})
    corp = {r.openalex_id: corpus_future_row(r) for r in df.itertuples(index=False)}
    rows = []
    for r in man.to_dict("records"):
        t = corp.get(r["future_id"]) or got.get(r["future_id"]) or {}
        rows.append({**r, **{k: t.get(k) for k in ("title", "abstract_raw", "abstract_clean", "abstract_norm")}})
    fut = pd.DataFrame(rows).drop(columns=["in_corpus", "sim_to_seed"], errors="ignore")
    print("restored text for", int(fut.abstract_norm.notna().sum()), "of", len(fut), "future rows")
    write_outputs(None, fut, X, row, write_manifests=False)


def main():
    df = pd.read_parquet(LOCAL / "works.parquet")
    ids, X = load_embeddings()
    row = {k: i for i, k in enumerate(ids)}
    if "--from-manifest" in sys.argv:
        return from_manifest(df, X, row)
    seeds, fut = build(df, X, row)
    seeds = add_placebos(seeds, X, row)
    fs = fut.groupby("seed_id").future_id.apply(set).to_dict()
    seeds["placebo_future_overlap"] = [round(len(fs[a] & fs[b]) / len(fs[a]), 3)
                                       for a, b in zip(seeds.seed_id, seeds.placebo_seed_id)]
    write_outputs(seeds, fut, X, row)
    print(seeds.groupby("journal").agg(n=("seed_id", "size"), topics=("primary_topic", "nunique"),
                                       med_fut=("n_citing_future_total", "median")))
    print("placebo match levels:", seeds.placebo_match.value_counts().to_dict())
    print(fut.groupby("kind").size().to_dict(), "OpenAlex", oa.usage_summary())


if __name__ == "__main__":
    main()
