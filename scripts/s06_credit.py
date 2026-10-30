"""Step 6: CREDIT inputs (IDs only; committed) -> credit/.

players.json        : 2016-2019 in_core works (with abstracts)
future_T.json       : 2021-2026 in_core works (with abstracts)
citation_edges.jsonl: one line per future work t: {"t_id", "t_journal", "t_year", "cited_players": [...]}
                      (from OpenAlex referenced_works; empty list if t cites no player)
credit_sets.csv     : id, set (player|future), journal, year, split -- convenience table
Embeddings for every player and t are in data/local/embeddings.npy (asserted here).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm.data import LOCAL, load_embeddings  # noqa: E402

OUT = ROOT / "credit"


def main():
    df = pd.read_parquet(LOCAL / "works.parquet")
    ids, _ = load_embeddings()
    core = df[df.in_core]
    players = core[core.year.between(2016, 2019)].sort_values(["journal", "year", "openalex_id"])
    future = core[core.year.between(2021, 2026)].sort_values(["journal", "year", "openalex_id"])
    assert set(players.openalex_id) | set(future.openalex_id) <= set(ids), "embeddings missing for CREDIT works"
    pset = set(players.openalex_id)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "players.json").write_text(json.dumps(players.openalex_id.tolist()))
    (OUT / "future_T.json").write_text(json.dumps(future.openalex_id.tolist()))
    n_edges, n_t_with = 0, 0
    with open(OUT / "citation_edges.jsonl", "w") as f:
        for r in future.itertuples(index=False):
            cited = sorted(set(r.referenced_works) & pset)
            n_edges += len(cited)
            n_t_with += bool(cited)
            f.write(json.dumps({"t_id": r.openalex_id, "t_journal": r.journal, "t_year": int(r.year),
                                "n_refs": len(r.referenced_works), "cited_players": cited}) + "\n")
    tab = pd.concat([players.assign(set="player"), future.assign(set="future")])[
        ["openalex_id", "set", "journal", "year", "split"]]
    tab.to_csv(OUT / "credit_sets.csv", index=False)
    indeg = pd.Series([p for line in open(OUT / "citation_edges.jsonl") for p in json.loads(line)["cited_players"]]).value_counts()
    stats = {"n_players": len(players), "n_future": len(future), "players_by_journal": players.journal.value_counts().to_dict(),
             "future_by_journal": future.journal.value_counts().to_dict(), "n_edges": n_edges,
             "n_future_citing_any_player": n_t_with, "share_future_citing_any_player": round(n_t_with / len(future), 3),
             "n_players_cited_at_least_once": int(len(indeg)), "share_players_cited": round(len(indeg) / len(players), 3),
             "future_with_no_refs": int((future.referenced_works.map(len) == 0).sum())}
    (ROOT / "data" / "manifest" / "credit_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
