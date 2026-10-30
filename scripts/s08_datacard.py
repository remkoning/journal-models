"""Step 8: write data/README.md (data card) from the built artifacts and stats files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jm import openalex as oa  # noqa: E402
from jm.data import LOCAL  # noqa: E402

M = ROOT / "data" / "manifest"


def md(df: pd.DataFrame) -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else (str(int(v)) if isinstance(v, float) and v.is_integer()
                                                           and abs(v) >= 2 else str(v)) for v in r) + " |")
    return "\n".join(lines)


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main():
    df = pd.read_parquet(LOCAL / "works.parquet")
    audit = json.loads((M / "cleaning_audit.json").read_text())
    ps = json.loads((M / "pair_stats.json").read_text())
    cs = json.loads((M / "credit_stats.json").read_text())
    em = json.loads((M / "embedding_meta.json").read_text())
    seeds = pd.DataFrame(jl(ROOT / "develop" / "seeds.jsonl"))
    fut = pd.DataFrame(jl(ROOT / "develop" / "future_manifest.jsonl"))
    gong = pd.DataFrame(jl(ROOT / "evals" / "external_gong_manifest.jsonl"))
    usage = oa.usage_summary()
    lt = json.loads((M / "lower_tier_stats.json").read_text()) if (M / "lower_tier_stats.json").exists() else None
    cal = pd.DataFrame(jl(ROOT / "evals" / "calibration_manifest.jsonl")) if lt else None
    calib_md = ""
    if lt:
        lt_counts = pd.DataFrame(lt["counts"]).T
        lt_counts["total"] = lt_counts.sum(axis=1)
        lt_split = pd.DataFrame(lt["split_counts"]).T[["train", "val", "test"]]
        ov = pd.DataFrame(lt["topical_overlap"]).T.drop(columns=["top_subfields"], errors="ignore").reset_index(names="journal")
        au = pd.DataFrame(lt["audit_clean_pct"]).T.reset_index(names="journal")
        cc = cal.groupby(["source", "tier"]).size().reset_index(name="n")
        cy = pd.DataFrame(lt["calibration_years"]).T.reset_index(names="source")
        drops_l = pd.Series(lt["dropped"]).rename_axis("journal / reason").reset_index(name="n")
        drops_l["journal / reason"] = drops_l["journal / reason"].str.replace("|", " / ", regex=False)
        calib_md = f"""
## CALIBRATION: "is my paper good enough for journal J?"

**Lower-tier journals.** The two are chosen from Gong, Li & Zhou's management tier map (ST3, in
`data/local/external/ai_taste_3e6637c7/`). The requirements were: non-Elsevier, ≥85% OpenAlex abstract coverage for
2022-2026 articles, and topical overlap with OrgSci/SMJ.

| short | journal | publisher | Gong tier | OpenAlex source | abstract coverage 2022-26 |
|---|---|---|---|---|---|
| JMS | Journal of Management Studies | Wiley | Fair | S151705444 | 98.2% |
| GOM | Group & Organization Management | SAGE | Limited | S192266589 | 87.8% |

JMS has the closest strategy, organization-theory, innovation and entrepreneurship overlap among the Fair journals.
GOM, from the Limited tier, adds a second step down, mostly on organizations and teams.

Excluded candidates and why:
* Leadership Quarterly is Elsevier.
* Human Relations is Fair, with 93.7% coverage; it is the fallback if a second Fair journal is wanted.
* Human Resource Management, Journal of Organizational Behavior, J Business & Psychology, J Managerial Psychology,
  J Org Behavior Management and J Personnel Psychology are HR- or psychology-centred.

Topical overlap was checked against the OrgSci+SMJ 2022-2026 corpus by nearest-neighbour bge cosine. Both lower-tier
journals are about as close to the flagships as the flagships are to each other:

{md(ov)}

**Table.** `data/local/lower_tier.parquet` has the same columns as `works.parquet`, plus `tier="lower"` and `gong_tier`.
It holds 2022-2026 articles with abstracts and is cleaned by the same pipeline: same filters, `abstract_norm` ≤180
words, and the journals' own names masked as `[JOURNAL]`. Embeddings are in `lower_tier_embeddings.npy` and
`lower_tier_embedding_ids.json` (same bge model). `split` uses the same ID hash. In this table, `in_core=False` means
"not applicable". Nothing was added to `works.parquet`, and no existing split, pair or ID file was changed.
`jm.data.text(id)` resolves these IDs, and `load_lower_tier()` loads the table.

{md(lt_counts.reset_index(names="journal"))}

{md(lt_split.reset_index(names="journal"))}

Dropped rows (same filters as the main corpus):

{md(drops_l)}

Residual giveaway audit on the clean text (%); `own_name` is JMS/GOM name mentions after masking:

{md(au)}

**Probe set: `evals/calibration_manifest.jsonl`** (IDs only; fields `openalex_id, source, tier, year`). All rows are
from the test split and 2022-2026. It has 20 core papers each from OrgSci, SMJ and MgmtSci, and 30 each from JMS and
GOM, so 60 are `top` and 60 `lower`. Each source has equal per-year quotas (4 per year for the 20s, 6 per year for the
30s). Gong external-check items are excluded. Load it with `jm.data.load_calibration()`.

{md(cc)}

{md(cy)}
"""
    fetched = sorted({json.loads(x)["ts"][:10] for x in oa.USAGE_LOG.read_text().splitlines()}) if oa.USAGE_LOG.exists() else []

    # counts
    jy = df.pivot_table(index="year", columns="journal", values="openalex_id", aggfunc="size", fill_value=0)
    jy_core = df[df.in_core].pivot_table(index="year", columns="journal", values="openalex_id", aggfunc="size", fill_value=0)
    jy = jy[["OrgSci", "SMJ", "MgmtSci"]].join(jy_core[["MgmtSci"]].rename(columns={"MgmtSci": "MgmtSci in_core"}))
    jy.loc["total"] = jy.sum()
    js = df.groupby(["journal", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    js_core = df[df.in_core].groupby(["journal", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    js = js.join(js_core.add_suffix(" (in_core)")).reindex(["OrgSci", "SMJ", "MgmtSci"])
    js.insert(0, "in_core", df[df.in_core].groupby("journal").size())
    js.insert(0, "works", df.groupby("journal").size())

    ms = df[df.journal == "MgmtSci"]
    dept = ms.ms_department.fillna("(no 'accepted by' line)").value_counts().rename_axis("ms_department").reset_index(name="n")
    dept["in_core"] = dept.ms_department.isin(ms[ms.in_core].ms_department.unique())

    pats = list(audit["clean"]["OrgSci"].keys())
    pats.remove("n")
    arows = []
    for p in pats:
        r = {"pattern": p}
        for j in ["OrgSci", "SMJ", "MgmtSci"]:
            r[f"{j} raw %"] = audit["raw"][j][p]
            r[f"{j} clean %"] = audit["clean"][j][p]
        arows.append(r)
    words = pd.DataFrame(audit["n_words_median"]).T.reset_index(names="journal")
    drops = pd.DataFrame([{"journal": k.split("|")[0], "reason": k.split("|")[1], "n": v} for k, v in audit["dropped"].items()])
    drops = drops.pivot_table(index="reason", columns="journal", values="n", fill_value=0)[["OrgSci", "SMJ", "MgmtSci"]].reset_index()

    pc = []
    for kind, key in (("cross", "cross_counts"), ("cross_fullms", "cross_fullms_counts"), ("cite", "cite_counts")):
        for k, v in ps[key].items():
            g, s = k.split("|")
            pc.append({"file": kind, "group": g, "split": s, "n": v})
    pc = pd.DataFrame(pc).pivot_table(index=["file", "group"], columns="split", values="n", fill_value=0)[["train", "val", "test"]]
    pc["total"] = pc.sum(axis=1)
    pc = pc.reset_index()
    cross_sim = pd.DataFrame(ps["cross_sim"]).T.reset_index(names="journal_pair")
    unb = pd.DataFrame(ps["cross_unblocked_threshold_table"])

    sd = seeds.groupby("journal").agg(seeds=("seed_id", "size"), topics=("primary_topic", "nunique"),
                                      years=("year", lambda x: f"{x.min()}-{x.max()}"),
                                      median_future_citers=("n_citing_future_total", "median"),
                                      mean_placebo_overlap=("placebo_future_overlap", "mean")).round(3).reset_index()
    fk = fut.groupby("kind").agg(rows=("future_id", "size"), unique_works=("future_id", "nunique"),
                                 mean_sim_to_seed=("sim_to_seed", "mean")).round(3).reset_index()

    text = f"""# Data card: journal-models corpus and eval sets

Built {em.get('built')} (OpenAlex queried {', '.join(fetched) or 'n/a'}). Generated by `scripts/s08_datacard.py`; numbers below
come from the committed manifests and `data/manifest/*.json`.

**Copyright rule.** Abstract text is never committed. All text lives in `data/local/` (gitignored). The repo holds IDs,
labels, pairs and metadata only (`data/manifest/`, `evals/`, `develop/`, `credit/`). Titles appear in
`works_manifest.csv` as bibliographic metadata (OpenAlex, CC0).

## Sources

* **OpenAlex** works API (`api.openalex.org`, polite pool `mailto=remkoning@gmail.com`, no API key). Filter per
  journal-year: `primary_location.source.id:<S>,publication_year:<y>,has_abstract:true`, 200 per page, 2016-2026.
  Sources: Organization Science `S206124708`, Strategic Management Journal `S102949365`, Management Science `S33323087`.
  Abstracts are reconstructed from `abstract_inverted_index`. Citation counts come from `counts_by_year`, which is
  current to the query date, so 2026 is a partial year.
  OpenAlex usage for this build: {usage['calls']} calls, {usage['credits']} credits (free tier: 1,000 credits/day without a key;
  cursor paging needs a key, so we use page-based paging per journal-year). Full log: `data/local/openalex_usage.jsonl`.
* **DEVELOP citing works** come from OpenAlex `filter=cites:<seed>,publication_year:>seed_year+1,has_abstract:true,
  type:article|review|book-chapter`, sorted by `cited_by_count:desc`, top 10, any venue.
* **External check**: Gong, Li & Zhou (2026), `github.com/FutureTech-OB/ai-taste` at commit `3e6637c7`. Data are CC BY 4.0.
  Only data files were downloaded; no code from the repo was run.
* **Embeddings**: `{em['model']}` (HF revision `{em.get('hf_revision')}`), {em['dim']}-d, L2-normalized, no query instruction.
  Built locally with sentence-transformers {em.get('sentence_transformers')} and torch {em.get('torch')}, on the `{em['text_field']}` field.
  Cost: $0.

## Corpus: `data/local/works.parquet`

One row per research article; {len(df):,} works in total. Columns: `openalex_id` (short id, e.g. `W2963...`), `doi`, `journal`
(`OrgSci`|`SMJ`|`MgmtSci`), `year`, `publication_date`, `type`, `title`, `abstract_raw`, `abstract_clean`, `abstract_norm`,
`n_words` (of clean), `n_words_norm`, `ms_department` (MgmtSci only), `in_core`, `split`, `cites_total`, `cites_5yr`
(sum of `counts_by_year` from pub year through pub year+5; null when that window runs past 2025, i.e. for years 2021 and later), `counts_by_year`
(JSON), `primary_topic`, `primary_topic_id`, `subfield`, `field`, `referenced_works` (list of short ids).

Load it with `jm.data.load_works()` and `jm.data.text(id)`. The ID and metadata copy is `data/manifest/works_manifest.csv`.

### Works per journal and year (with abstracts, after cleaning and filters)

{md(jy.reset_index())}

### Works per journal and split

The split is assigned per paper by a stable hash: `sha256(short_openalex_id) mod 100`. Values below 50 go to train, 50-69 to val, and 70-99 to test.

{md(js.reset_index())}

### `in_core` and the MgmtSci department breakdown

`in_core` is True for every OrgSci and SMJ article. For MgmtSci it is True only when the department parsed from the "This paper
was accepted by <editor>, <department>." line is business strategy, strategy, organizations, or entrepreneurship and
innovation. The pattern also accepts technology/innovation-management variants, but none occur in 2016-2026.
MgmtSci papers without an "accepted by" line ({int(ms.ms_department.isna().sum())} papers) cannot be classified, so they are
not in_core. The full MgmtSci set is kept in `works.parquet` for robustness checks.

{md(dept)}

## Cleaning (`abstract_clean`, `abstract_norm`)

Rules, in `jm/clean.py`:
1. Parse the MgmtSci department from the "accepted by" sentence, then strip that sentence. "Last, First, dept" name
   forms and special-issue/section variants are handled.
2. Cut labelled INFORMS tail sections (`Funding:`, `Supplemental Material:`, `History:`, `Open Access Statement:`,
   `Conflict of Interest:`, `Data Ethics & Reproducibility Note:`, ...).
3. Drop sentences that point to the online appendix, e-companion, data files or replication files, or that contain
   a 10.1287 or 10.1002 DOI.
4. SMJ: keep only the Research Summary. Drop the `Abstract` prefix, the `Research Summary` header and the
   `Managerial Summary`/`Managerial Abstract` text.
5. Strip copyright and publisher tails (`©`, `Copyright 20xx John Wiley & Sons`, Creative Commons, published by INFORMS).
6. Replace the remaining literal journal names with `[JOURNAL]`. This hits 19 works, all genuine mentions such as
   "a paper published in Management Science".

`abstract_norm` is `abstract_clean` truncated at a sentence boundary to at most 180 words.

**Filters.** These works are dropped and logged with reasons in `data/manifest/dropped_works.csv`:
* DOI prefix does not match the journal. This catches 147 misattributed records under MgmtSci.
* OpenAlex type is editorial, erratum, letter, review or other. Paratext and conference-abstract records are kept only if
  they pass the content checks.
* Retracted works.
* Non-research titles or abstracts: errata, corrections, editorials, introductions, comments, replies, commentaries,
  awards, calls for papers, reviewer acknowledgements, and the "SMJ seeks to publish" boilerplate.
* Clean abstracts under 50 words.
* Duplicate titles.

{md(drops)}

### Residual giveaway audit (% of kept works matching each pattern)

`accepted_by` residues are genuine prose ("accepted by workers", "accepted by the government"), checked by hand.

{md(pd.DataFrame(arows))}

Median abstract length in words:

{md(words)}

SMJ research summaries are shorter than OrgSci and MgmtSci abstracts even after truncation, so length is still a weak
cue, now in the opposite direction from the raw text.

## EVALUATE pairs: `evals/pairs/` (IDs only)

Pairs are formed only between papers in the same split and use in_core papers only.

**`cross_{{split}}.jsonl`** holds topic-matched cross-journal pairs. Rule: {ps['cross_rule']}. Threshold = {ps['cross_threshold']},
HI_SIM = {ps['hi_sim_override']}. Each paper is used at most once per journal pair, and which paper is "a" is randomized (seeded).
Fields: `pair_id, a_id, b_id, a_journal, b_journal, a_year, b_year, sim, same_topic, same_primary_topic, same_subfield,
journal_pair, split`. `same_topic` = same primary topic OR same subfield.

*Why the subfield block?* OpenAlex topic labels are noisy compared with embedding similarity. With plain cosine thresholding
and no block, the share of matched pairs that share a topic or subfield never reaches 70% at usable N. It peaks at about
40% at cosine 0.80, and only about 10 pairs survive at 0.88. On inspection, unblocked pairs at 0.80 and above are clearly
topical, and pairs below 0.75 are not. So we require the same OpenAlex subfield (or cosine >= HI_SIM) and set a cosine
floor of 0.75. The unblocked diagnostic table:

{md(unb)}

Realized similarity and topic agreement of the final cross pairs:

{md(cross_sim)}

**`cite_{{split}}.jsonl`** holds within-journal "promise" pairs. Same journal and same year (2016-2020), in_core only. One paper is
in the top quartile and the other in the bottom quartile of `cites_5yr` within that journal-year. Quartiles are computed
over in_core papers of that journal-year across all splits. Each high paper is matched to an unused low paper in the
same split, preferring the same subfield and then the highest cosine. Order is randomized. Fields: `pair_id, a_id, b_id,
journal, year, a_cites5, b_cites5, higher, same_subfield, sim, q25, q75, split`.
Share with the same subfield: {ps['cite_share_same_subfield']}. Share with `higher == "a"`: {ps['cite_share_higher_a']}. Median hi/lo citation ratio: {ps['cite_median_ratio_hi_lo']}.

**`cross_fullms_{{split}}.jsonl`** is a robustness set. It uses the same rule, but the MgmtSci side draws on all MgmtSci
departments; only the two MgmtSci journal pairs are included.

{md(pc)}

## DEVELOP: `develop/seeds.jsonl`, `develop/future_manifest.jsonl` (IDs only), `data/local/develop_future.parquet` (text)

Seeds are drawn per journal from 2016-2018, in_core, val/test split papers with at least 5 citing works published in year+2
or later (with abstracts). They are stratified across OpenAlex primary topics: topics are drawn proportional to size
without replacement, one paper per topic per round. MgmtSci has only 27 eligible papers, so it gets 26 seeds, not 30.

{md(sd)}

Future rows per seed:
* `citing_top`: up to 10 of the most-cited citing works, any venue, published in year+2 or later.
* `semantic_neighbor`: the 20 nearest 2021-2026 works in our 3-journal corpus (all departments) that do **not** cite
  the seed, by cosine.
* `semantic_citer`: citing works met along the way, with `cites_seed=True` and their rank.

`sim_to_seed` is the bge cosine. Out-of-corpus citing works are embedded in `data/local/develop_future_embeddings.npy`.

{md(fk)}

**Placebo.** `placebo_seed_id` is another seed with the same primary topic (else the same subfield, else any seed), chosen as
the nearest by embedding. The match level is in `placebo_match`. `placebo_future_overlap` is the share of the seed's future
IDs that also appear in the placebo's future. Downstream scoring should drop the shared IDs.

## CREDIT: `credit/` (IDs only)

* `players.json`: {cs['n_players']} in_core works from 2016-2019 ({cs['players_by_journal']}).
* `future_T.json`: {cs['n_future']} in_core works from 2021-2026 ({cs['future_by_journal']}).
* `citation_edges.jsonl`: one line per t, listing the players it cites (from OpenAlex `referenced_works`). There are
  {cs['n_edges']} edges. {cs['share_future_citing_any_player']:.0%} of the t cite at least one player, and
  {cs['share_players_cited']:.0%} of players are cited at least once. {cs['future_with_no_refs']} t have no references in OpenAlex.
* `credit_sets.csv` lists id, set, journal, year and split.

All players and all t have embeddings in `data/local/embeddings.npy`.

## External check: `evals/external_gong_manifest.jsonl` and `data/local/external_gong.jsonl`

The source is the Gong et al. management benchmark (120 items). We keep all 12 Organization Science and Strategic Management
Journal items, all labelled `top`, which maps to unified `exceptional`. We add 12 lowest-tier items (`fair`, unified
`limited`) from other management journals, chosen round-robin by journal. The text field used is **`rq_with_context`**,
the released research-idea pitch, which is not the abstract. All {int(gong.in_our_corpus_openalex_id.notna().sum())} OrgSci/SMJ items
are also in our corpus (2025 papers; their splits are recorded in `our_split`). Exclude them from any playbook training
before using this as an external check. The raw files are in `data/local/external/ai_taste_3e6637c7/`.

{md(gong.groupby(['selected_as', 'journal_short', 'unified_label']).size().reset_index(name='n'))}

{calib_md}
## Regenerate

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m data.build_all                  # restore data/local for the committed IDs/pairs/seeds
.venv/bin/python -m data.build_all --rebuild-evals  # re-derive every manifest/pair/seed from a fresh OpenAlex pull
```

The default mode pins the corpus to `data/manifest/works_manifest.csv` and leaves committed artifacts untouched. It
re-fetches DEVELOP citing works by ID. A cold run uses about 90 OpenAlex credits and about 3 minutes of local embedding.
OpenAlex is a moving target: citation counts and topics drift, so `--rebuild-evals` yields slightly different sets.
Responses are cached in `data/local/raw/openalex/`, so reruns on the same machine are free.

## Known limitations

* MgmtSci in_core is small: {int(df[(df.journal == 'MgmtSci') & df.in_core].shape[0])} papers. That caps MgmtSci cross pairs at about 34 test pairs
  per journal pair (target 150) and MgmtSci cite pairs at {int(sum(v for k, v in ps['cite_counts'].items() if k.startswith('MgmtSci')))} in total.
  Use `cross_fullms_*` for power.
* OrgSci-SMJ test cross pairs number {ps['cross_counts'].get('OrgSci-SMJ|test')} (target 150), limited by the 0.75 floor and 1:1 matching.
* `cites_5yr` covers 2016-2020 only, because of the fixed window.
* OpenAlex abstract coverage dips for MgmtSci in 2018: 133 of 237 articles have abstracts. Papers without abstracts are
  absent from the corpus.
* OpenAlex `primary_topic`/`subfield` are model-assigned and noisy. Use embeddings for similarity.
"""
    (ROOT / "data" / "README.md").write_text(text)
    print("wrote data/README.md", len(text), "chars")


if __name__ == "__main__":
    main()
