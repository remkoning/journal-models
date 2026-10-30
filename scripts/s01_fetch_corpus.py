"""Step 1: fetch all 2016-2026 works with abstracts from OrgSci, SMJ, MgmtSci (OpenAlex, cursor paging).

Writes data/local/raw/corpus_{journal}.jsonl (one OpenAlex work per line). ~36 credits total; cached.
Type filtering (drop editorials/errata/paratext) happens in step 2, so we can audit what was dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jm import openalex as oa  # noqa: E402

YEARS = range(2016, 2027)
SELECT = ("id,doi,title,publication_year,publication_date,type,is_retracted,abstract_inverted_index,"
          "cited_by_count,counts_by_year,primary_topic,referenced_works")


def main():
    out_dir = oa.LOCAL / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, sid in oa.SOURCES.items():
        works = []
        for y in YEARS:  # per journal-year keeps most queries to one page (stable paging without a key)
            works += oa.list_all(f"primary_location.source.id:{sid},publication_year:{y},has_abstract:true", SELECT)
        with open(out_dir / f"corpus_{j}.jsonl", "w") as f:
            for w in works:
                f.write(json.dumps(w) + "\n")
        print(j, len(works), flush=True)
    print("OpenAlex usage so far:", oa.usage_summary())


if __name__ == "__main__":
    main()
