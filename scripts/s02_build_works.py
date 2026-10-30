"""Step 2: build data/local/works.parquet (+ data/manifest/works_manifest.csv, cleaning audit, drop log).

--pin: keep only IDs already in data/manifest/works_manifest.csv and do not overwrite committed manifest files
       (used by data.build_all to restore the local store for the committed eval sets).
Filters: journal DOI prefix, type, retractions, non-research titles/abstracts (editorials, errata, comments,
replies, awards, calls for papers), abstracts < 50 words after cleaning, duplicate titles.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm import clean as C  # noqa: E402
from jm.openalex import LOCAL, abstract_from_inv, short_id  # noqa: E402

MANIFEST = ROOT / "data" / "manifest"
JOURNALS = ["OrgSci", "SMJ", "MgmtSci"]
LAST_COMPLETE_YEAR = 2025   # citations through 2025 are complete as of the 2026-10-30 pull
MIN_WORDS = 50

# MgmtSci departments that define the strategy/organizations/entrepreneurship core.
CORE_DEPT_PATTERNS = re.compile(
    r"^(?:business strategy|strategy|strategic management|organizations?|organization(?:al)? (?:behavior|theory)|"
    r"entrepreneurship(?: (?:and|&) innovation)?|innovation(?: (?:and|&) entrepreneurship)?|"
    r"technology,? innovation,? and entrepreneurship|technological innovation,? product development,? and entrepreneurship|"
    r"technology (?:and|&) innovation management|management of technology|organizations and strategy)$")


def split_of(oa_id: str) -> str:
    """Stable paper-level split: sha256 of the short OpenAlex id (e.g. 'W2341'), mod 100."""
    b = int(hashlib.sha256(short_id(oa_id).encode()).hexdigest()[:8], 16) % 100
    return "train" if b < 50 else ("val" if b < 70 else "test")


def cites_window(cby: list[dict], y0: int, horizon: int = 5) -> float | None:
    if y0 + horizon > LAST_COMPLETE_YEAR:
        return None
    return float(sum(c["cited_by_count"] for c in cby if y0 <= c["year"] <= y0 + horizon))


def audit_rates(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for j, g in df.groupby("journal"):
        out[j] = {k: round(float(g[col].str.contains(rx).mean()) * 100, 2) for k, rx in C.AUDIT_RE.items()}
        out[j]["n"] = int(len(g))
    return out


def main():
    rows, dropped = [], []
    for j in JOURNALS:
        for line in open(LOCAL / "raw" / f"corpus_{j}.jsonl"):
            w = json.loads(line)
            wid = short_id(w["id"])
            doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            title = (w.get("title") or "").strip()
            raw = abstract_from_inv(w.get("abstract_inverted_index"))
            reason = None
            if not doi.startswith(C.DOI_PREFIX[j]):
                reason = "doi_prefix_mismatch"
            elif w.get("type") not in C.KEEP_TYPES:
                reason = f"type:{w.get('type')}"
            elif w.get("is_retracted"):
                reason = "retracted"
            elif not title or C.NONRESEARCH_TITLE.search(title):
                reason = "nonresearch_title"
            elif C.NONRESEARCH_ABSTRACT.search(raw):
                reason = "nonresearch_abstract"
            clean, dept = C.clean_abstract(raw, j)
            if reason is None and len(clean.split()) < MIN_WORDS:
                reason = "short_abstract"
            if reason:
                dropped.append({"openalex_id": wid, "journal": j, "year": w["publication_year"], "type": w.get("type"),
                                "doi": doi, "title": title, "reason": reason})
                continue
            pt = w.get("primary_topic") or {}
            cby = sorted(w.get("counts_by_year") or [], key=lambda c: c["year"])
            rows.append({
                "openalex_id": wid, "doi": doi, "journal": j, "year": int(w["publication_year"]),
                "publication_date": w.get("publication_date"), "type": w.get("type"), "title": title,
                "abstract_raw": raw, "abstract_clean": clean, "ms_department": dept if j == "MgmtSci" else None,
                "cites_total": int(w.get("cited_by_count") or 0),
                "cites_5yr": cites_window(cby, int(w["publication_year"])),
                "counts_by_year": json.dumps({str(c["year"]): c["cited_by_count"] for c in cby}),
                "primary_topic": pt.get("display_name"), "primary_topic_id": short_id(pt["id"]) if pt.get("id") else None,
                "subfield": (pt.get("subfield") or {}).get("display_name"),
                "field": (pt.get("field") or {}).get("display_name"),
                "referenced_works": [short_id(r) for r in (w.get("referenced_works") or [])],
            })
    df = pd.DataFrame(rows)
    # duplicates: same journal + normalized title -> keep the more-cited record
    df["_t"] = df["title"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    df = df.sort_values(["cites_total"], ascending=False)
    dup = df.duplicated(["journal", "_t"], keep="first")
    for _, r in df[dup].iterrows():
        dropped.append({"openalex_id": r.openalex_id, "journal": r.journal, "year": r.year, "type": r.type,
                        "doi": r.doi, "title": r.title, "reason": "duplicate_title"})
    df = df[~dup].drop(columns="_t").sort_values(["journal", "year", "openalex_id"]).reset_index(drop=True)
    pin = "--pin" in sys.argv and (MANIFEST / "works_manifest.csv").exists()
    if pin:
        keep = set(pd.read_csv(MANIFEST / "works_manifest.csv", usecols=["openalex_id"]).openalex_id)
        missing = keep - set(df.openalex_id)
        df = df[df.openalex_id.isin(keep)].reset_index(drop=True)
        print(f"pinned to manifest: {len(df)} works; {len(missing)} manifest IDs not in this pull")

    df["abstract_norm"] = df["abstract_clean"].map(lambda s: C.normalize_length(s, 180))
    df["n_words"] = df["abstract_clean"].str.split().str.len()
    df["n_words_norm"] = df["abstract_norm"].str.split().str.len()
    dept = df["ms_department"].fillna("")
    df["in_core"] = (df["journal"] != "MgmtSci") | dept.str.match(CORE_DEPT_PATTERNS)
    df["split"] = df["openalex_id"].map(split_of)
    cols = ["openalex_id", "doi", "journal", "year", "publication_date", "type", "title", "abstract_raw", "abstract_clean",
            "abstract_norm", "n_words", "n_words_norm", "ms_department", "in_core", "split", "cites_total", "cites_5yr",
            "counts_by_year", "primary_topic", "primary_topic_id", "subfield", "field", "referenced_works"]
    df = df[cols]
    df.to_parquet(LOCAL / "works.parquet", index=False)

    mdir = (LOCAL / "manifest_rebuild") if pin else MANIFEST
    mdir.mkdir(parents=True, exist_ok=True)
    man = df.drop(columns=["abstract_raw", "abstract_clean", "abstract_norm", "referenced_works", "counts_by_year"])
    man.to_csv(mdir / "works_manifest.csv", index=False)
    pd.DataFrame(dropped).sort_values(["journal", "year", "openalex_id"]).to_csv(mdir / "dropped_works.csv", index=False)

    audit = {"raw": audit_rates(df, "abstract_raw"), "clean": audit_rates(df, "abstract_clean"),
             "norm": audit_rates(df, "abstract_norm"),
             "n_words_median": {j: {"raw": float(g["abstract_raw"].str.split().str.len().median()),
                                    "clean": float(g["n_words"].median()), "norm": float(g["n_words_norm"].median()),
                                    "share_truncated": round(float((g["n_words"] > 180).mean()), 3)}
                                for j, g in df.groupby("journal")},
             "dropped": Counter(f"{d['journal']}|{d['reason'].split(':')[0]}" for d in dropped)}
    (mdir / "cleaning_audit.json").write_text(json.dumps(audit, indent=1, default=int))

    print("kept", len(df), "dropped", len(dropped))
    print(df.groupby("journal").agg(n=("openalex_id", "size"), core=("in_core", "sum")))
    print("drop reasons:", dict(audit["dropped"]))
    print("MgmtSci departments:\n", df[df.journal == "MgmtSci"]["ms_department"].fillna("<none>").value_counts().to_string())
    print(json.dumps({"clean": audit["clean"], "words": audit["n_words_median"]}, indent=1))


if __name__ == "__main__":
    main()
