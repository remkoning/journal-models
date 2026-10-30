"""Step 7: external check data from Gong, Li & Zhou (2026), github.com/FutureTech-OB/ai-taste (data CC BY 4.0).

Downloads (data files only; no code is executed) into data/local/external/ai_taste_<sha8>/, pinned to COMMIT.
Keeps: all items from Organization Science or Strategic Management Journal, plus N_LOWER lowest-tier ('fair' ->
unified 'limited') management items, stratified by journal. Text field used: `rq_with_context` (the released
research-idea pitch that the paper uses as evaluation input).
Outputs: data/local/external_gong.jsonl (with text), evals/external_gong_manifest.jsonl (no text).
"""
from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm.data import LOCAL  # noqa: E402

REPO = "FutureTech-OB/ai-taste"
COMMIT = "3e6637c7b8faec0977185185c0bd315c58a9eba2"
FILES = ["LICENSE", "README.md", "data/README.md", "data/management_deep_probe/README.md",
         "data/management_deep_probe/benchmark/README.md", "data/management_deep_probe/benchmark/management_benchmark_120.jsonl",
         "data/supplementary_tables/ST18_label_normalization.csv"]
BENCH = "data/management_deep_probe/benchmark/management_benchmark_120.jsonl"
TARGET = {"ORGANIZATION SCIENCE": "OrgSci", "STRATEGIC MANAGEMENT JOURNAL": "SMJ"}
N_LOWER = 12
TEXT_FIELD = "rq_with_context"
DEST = LOCAL / "external" / f"ai_taste_{COMMIT[:8]}"


def local_name(path: str) -> Path:
    return DEST / path.replace("/", "_")


def fetch():
    DEST.mkdir(parents=True, exist_ok=True)
    for p in FILES:
        out = local_name(p)
        if not out.exists():
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{p}"
            with urllib.request.urlopen(url, timeout=60) as r:
                out.write_bytes(r.read())


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def main():
    fetch()
    labels = {r["source_label"]: r["unified_label"] for r in csv.DictReader(open(local_name(FILES[-1])))
              if r["source_label_space"].startswith("Article ground truth")}
    rows = [json.loads(line) for line in open(local_name(BENCH)) if line.strip()]
    for i, r in enumerate(rows):
        r["gong_idx"] = i
        r["unified_label"] = labels.get(r.get("level"), r.get("level"))
    sel = [dict(r, selected_as="target_journal", journal_short=TARGET[r["journal"]]) for r in rows if r["journal"] in TARGET]
    lower = pd.DataFrame([r for r in rows if r["unified_label"] == "limited" and r["journal"] not in TARGET])
    rng = np.random.default_rng(20261030)
    lower = lower.assign(_r=rng.random(len(lower))).sort_values(["_r"])
    lower["_k"] = lower.groupby("journal").cumcount()            # round-robin across journals
    lower = lower.sort_values(["_k", "_r"]).head(N_LOWER).drop(columns=["_r", "_k"])
    sel += [dict(r, selected_as="lower_tier", journal_short="other") for r in lower.to_dict("records")]

    works = pd.read_parquet(LOCAL / "works.parquet", columns=["openalex_id", "title", "journal", "year", "split"])
    tmap = dict(zip(works.title.map(norm_title), works.openalex_id))
    wsplit = dict(zip(works.openalex_id, works.split))
    wyear = dict(zip(works.openalex_id, works.year))
    out_local, out_man = [], []
    for r in sel:
        oa_id = tmap.get(norm_title(r["title"]))
        rec = {"gong_idx": r["gong_idx"], "title": r["title"], "journal": r["journal"], "journal_short": r["journal_short"],
               "level": r["level"], "unified_label": r["unified_label"], "domain": r.get("domain"),
               "selected_as": r["selected_as"], "text_field": TEXT_FIELD, "in_our_corpus_openalex_id": oa_id,
               "our_split": wsplit.get(oa_id), "our_year": int(wyear[oa_id]) if oa_id else None,
               "source": f"github.com/{REPO}@{COMMIT[:8]}/{BENCH}", "license": "CC BY 4.0 (Gong, Li & Zhou 2026)"}
        out_man.append(rec)
        out_local.append({**rec, "text": r[TEXT_FIELD], "n_words": len(r[TEXT_FIELD].split())})
    with open(LOCAL / "external_gong.jsonl", "w") as f:
        for r in out_local:
            f.write(json.dumps(r) + "\n")
    (ROOT / "evals").mkdir(exist_ok=True)
    with open(ROOT / "evals" / "external_gong_manifest.jsonl", "w") as f:
        for r in out_man:
            f.write(json.dumps(r) + "\n")
    d = pd.DataFrame(out_man)
    print(d.groupby(["selected_as", "journal", "unified_label"]).size().to_string())
    print("matched to our corpus:", d.in_our_corpus_openalex_id.notna().sum(), "of", len(d))


if __name__ == "__main__":
    main()
