"""Step 9: CALIBRATION set for "is my paper good enough for journal J?".

Lower-tier journals (Gong, Li & Zhou 2026 management tier map, ST3; non-Elsevier; >=85% abstract coverage 2022-2026):
  JMS = Journal of Management Studies (Wiley; Gong tier 'Fair')       strategy/organization/innovation overlap
  GOM = Group & Organization Management (SAGE; Gong tier 'Limited')   organizations/teams overlap
Pulls 2022-2026 articles, cleans them with the same pipeline (+ masks their own journal names), embeds with the same
bge model. Writes data/local/lower_tier.parquet (+ lower_tier_embeddings.npy / _ids.json) -- separate from works.parquet;
no existing split/pair/ID file is touched.
Writes evals/calibration_manifest.jsonl (IDs only): test split, 2022-2026; 20 OrgSci + 20 SMJ + 20 MgmtSci (in_core)
+ 30 per lower-tier journal, with equal per-year quotas where possible. Gong external-check items are excluded.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm import clean as C  # noqa: E402
from jm import openalex as oa  # noqa: E402
from jm.openalex import LOCAL, abstract_from_inv, short_id  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from s01_fetch_corpus import SELECT  # noqa: E402
from s02_build_works import MIN_WORDS, split_of  # noqa: E402

LOWER = {  # short: (OpenAlex source id, DOI prefix, Gong management tier, full name)
    "JMS": ("S151705444", "10.1111/joms", "fair", "Journal of Management Studies"),
    "GOM": ("S192266589", "10.1177/10596011", "limited", "Group & Organization Management"),
}
YEARS = range(2022, 2027)
LOWER_NAMES = re.compile(r"\b(?:Journal of Management Studies|JMS|Group (?:&|and) Organization Management|GOM)\b")
QUOTA = {"OrgSci": 20, "SMJ": 20, "MgmtSci": 20, "JMS": 30, "GOM": 30}
SEED = 20261030


def build_lower() -> tuple[pd.DataFrame, list[dict]]:
    works = pd.read_parquet(LOCAL / "works.parquet")
    cols = list(works.columns)
    rows, dropped = [], []
    for j, (sid, prefix, gtier, _) in LOWER.items():
        raw_path = LOCAL / "raw" / f"lower_{j}.jsonl"
        ws = []
        for y in YEARS:
            ws += oa.list_all(f"primary_location.source.id:{sid},publication_year:{y},has_abstract:true", SELECT)
        with open(raw_path, "w") as f:
            for w in ws:
                f.write(json.dumps(w) + "\n")
        for w in ws:
            wid, doi = short_id(w["id"]), (w.get("doi") or "").replace("https://doi.org/", "").lower()
            title = (w.get("title") or "").strip()
            raw = abstract_from_inv(w.get("abstract_inverted_index"))
            reason = None
            if not doi.startswith(prefix):
                reason = "doi_prefix_mismatch"
            elif w.get("type") not in C.KEEP_TYPES:
                reason = f"type:{w.get('type')}"
            elif w.get("is_retracted"):
                reason = "retracted"
            elif not title or C.NONRESEARCH_TITLE.search(title):
                reason = "nonresearch_title"
            elif C.NONRESEARCH_ABSTRACT.search(raw):
                reason = "nonresearch_abstract"
            clean, _ = C.clean_abstract(raw, j, extra_names=LOWER_NAMES)
            if reason is None and len(clean.split()) < MIN_WORDS:
                reason = "short_abstract"
            if reason:
                dropped.append({"openalex_id": wid, "journal": j, "year": w["publication_year"], "reason": reason})
                continue
            pt = w.get("primary_topic") or {}
            cby = sorted(w.get("counts_by_year") or [], key=lambda c: c["year"])
            rows.append({"openalex_id": wid, "doi": doi, "journal": j, "year": int(w["publication_year"]),
                         "publication_date": w.get("publication_date"), "type": w.get("type"), "title": title,
                         "abstract_raw": raw, "abstract_clean": clean, "ms_department": None, "in_core": False,
                         "split": split_of(wid), "cites_total": int(w.get("cited_by_count") or 0), "cites_5yr": None,
                         "counts_by_year": json.dumps({str(c["year"]): c["cited_by_count"] for c in cby}),
                         "primary_topic": pt.get("display_name"),
                         "primary_topic_id": short_id(pt["id"]) if pt.get("id") else None,
                         "subfield": (pt.get("subfield") or {}).get("display_name"),
                         "field": (pt.get("field") or {}).get("display_name"),
                         "referenced_works": [short_id(r) for r in (w.get("referenced_works") or [])],
                         "gong_tier": gtier})
    df = pd.DataFrame(rows)
    df["_t"] = df.title.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    df = df.sort_values("cites_total", ascending=False).drop_duplicates(["journal", "_t"]).drop(columns="_t")
    df["abstract_norm"] = df.abstract_clean.map(lambda s: C.normalize_length(s, 180))
    df["n_words"] = df.abstract_clean.str.split().str.len()
    df["n_words_norm"] = df.abstract_norm.str.split().str.len()
    df["cites_5yr"] = df["cites_5yr"].astype("float64")
    df["tier"] = "lower"
    df = df.sort_values(["journal", "year", "openalex_id"]).reset_index(drop=True)
    return df[cols + ["tier", "gong_tier"]], dropped


def sample_calibration(works: pd.DataFrame, lower: pd.DataFrame) -> pd.DataFrame:
    gong = {r["in_our_corpus_openalex_id"] for r in map(json.loads, open(ROOT / "evals" / "external_gong_manifest.jsonl"))}
    top = works[works.in_core & (works.split == "test") & works.year.between(2022, 2026) & ~works.openalex_id.isin(gong)]
    low = lower[(lower.split == "test") & lower.year.between(2022, 2026)]
    pool = pd.concat([top.assign(tier="top"), low])
    out = []
    for src, q in QUOTA.items():
        g = pool[pool.journal == src]
        rng = np.random.default_rng(SEED + sum(map(ord, src)))
        g = g.assign(_r=rng.random(len(g))).sort_values("_r")
        g["_k"] = g.groupby("year").cumcount()          # round-robin over years -> equal per-year quotas
        out.append(g.sort_values(["_k", "_r"]).head(q))
    cal = pd.concat(out)
    return cal.assign(source=cal.journal)[["openalex_id", "source", "tier", "year"]].reset_index(drop=True)


def overlap_stats(lower: pd.DataFrame, Xl: np.ndarray) -> dict:
    """Topical overlap with OrgSci/SMJ: nearest-neighbour cosine to the OrgSci+SMJ 2022-2026 corpus."""
    works = pd.read_parquet(LOCAL / "works.parquet", columns=["openalex_id", "journal", "year", "subfield"])
    ids = json.loads((LOCAL / "embedding_ids.json").read_text())
    X = np.load(LOCAL / "embeddings.npy")
    row = {k: i for i, k in enumerate(ids)}
    ref = works[works.journal.isin(["OrgSci", "SMJ"]) & works.year.between(2022, 2026)]
    R = X[[row[i] for i in ref.openalex_id]]
    out = {}
    for j, g in lower.groupby("journal"):
        nn = (Xl[g.index.to_numpy()] @ R.T).max(axis=1)
        out[j] = {"n": len(g), "nn_cos_to_orgsci_smj_median": round(float(np.median(nn)), 3),
                  "share_nn_cos_ge_0.75": round(float((nn >= 0.75).mean()), 3),
                  "top_subfields": g.subfield.value_counts(normalize=True).head(4).round(3).to_dict()}
    # reference: OrgSci vs SMJ (leave-self-out not needed across journals)
    for j in ["OrgSci", "SMJ"]:
        a = ref[ref.journal == j]
        b = ref[ref.journal != j]
        nn = (X[[row[i] for i in a.openalex_id]] @ X[[row[i] for i in b.openalex_id]].T).max(axis=1)
        out[f"{j} (vs the other flagship)"] = {"n": len(a), "nn_cos_to_orgsci_smj_median": round(float(np.median(nn)), 3),
                                               "share_nn_cos_ge_0.75": round(float((nn >= 0.75).mean()), 3)}
    return out


def main():
    from jm import embed as E
    lower, dropped = build_lower()
    Xl = E.embed(lower.abstract_norm.tolist())
    lower.to_parquet(LOCAL / "lower_tier.parquet", index=False)
    np.save(LOCAL / "lower_tier_embeddings.npy", Xl)
    (LOCAL / "lower_tier_embedding_ids.json").write_text(json.dumps(lower.openalex_id.tolist()))

    works = pd.read_parquet(LOCAL / "works.parquet")
    assert not set(lower.openalex_id) & set(works.openalex_id)
    if "--keep-manifest" in sys.argv and (ROOT / "evals" / "calibration_manifest.jsonl").exists():
        print("kept committed evals/calibration_manifest.jsonl; rebuilt data/local/lower_tier.parquet + embeddings")
        return
    cal = sample_calibration(works, lower)
    with open(ROOT / "evals" / "calibration_manifest.jsonl", "w") as f:
        for r in cal.to_dict("records"):
            f.write(json.dumps({**r, "year": int(r["year"])}) + "\n")

    audit = {j: {k: round(float(g.abstract_clean.str.contains(rx).mean()) * 100, 2) for k, rx in C.AUDIT_RE.items()}
             | {"own_name": round(float(g.abstract_clean.str.contains(LOWER_NAMES).mean()) * 100, 2),
                "raw_own_name": round(float(g.abstract_raw.str.contains(LOWER_NAMES).mean()) * 100, 2),
                "median_words_raw": float(g.abstract_raw.str.split().str.len().median()),
                "median_words_norm": float(g.n_words_norm.median())}
             for j, g in lower.groupby("journal")}
    stats = {"journals": {k: {"openalex_source": v[0], "doi_prefix": v[1], "gong_tier": v[2], "name": v[3]}
                          for k, v in LOWER.items()},
             "counts": lower.groupby(["journal", "year"]).size().unstack(fill_value=0).to_dict("index"),
             "split_counts": lower.groupby(["journal", "split"]).size().unstack(fill_value=0).to_dict("index"),
             "dropped": pd.DataFrame(dropped).groupby(["journal", "reason"]).size().to_dict() if dropped else {},
             "audit_clean_pct": audit, "topical_overlap": overlap_stats(lower, Xl),
             "calibration_counts": cal.groupby(["source", "tier"]).size().to_dict(),
             "calibration_years": cal.groupby(["source", "year"]).size().unstack(fill_value=0).to_dict("index")}
    stats["dropped"] = {f"{k[0]}|{k[1]}": int(v) for k, v in stats["dropped"].items()}
    stats["calibration_counts"] = {f"{k[0]}|{k[1]}": int(v) for k, v in stats["calibration_counts"].items()}
    (ROOT / "data" / "manifest" / "lower_tier_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))
    print("OpenAlex", oa.usage_summary())


if __name__ == "__main__":
    main()
