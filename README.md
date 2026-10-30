# Journal Models (prototype)

Prototype accompanying the Organization Science special-issue proposal *"The Journal as Model"* (Koning & Myers).

Each journal-model is a base LLM paired with an evolved, human-readable **taste playbook** for that journal. It performs three functions on research abstracts:

1. **EVALUATE**: does this work fit the journal's taste, and does it show promise?
2. **DEVELOP**: how could this work make a bigger contribution to the journal's community?
3. **CREDIT**: Shapley attribution of later knowledge to earlier contributions.

Journals in v0: *Organization Science*, *Strategic Management Journal*, *Management Science*. Data: OpenAlex.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

LLM access goes through `jm/llm.py`. It supports three backends:

- **`cli`**: Claude Code headless (`claude -p`). This is the default for Claude models.
- **`api`**: the Anthropic API. Set `ANTHROPIC_API_KEY`.
- **`openrouter`**: for non-Claude judges. Set `OPENROUTER_API_KEY`.

Every call is cached and logged to `results/ledger.jsonl`. Billed backends are capped by `JM_CAP_API_USD` and `JM_CAP_OPENROUTER_USD`.

## Data

All data are built from OpenAlex, with no LLM calls. Abstract text lives only in the gitignored `data/local/`; the repo
commits IDs, labels and pairs. Rebuild the local store with one command:

```bash
.venv/bin/python -m data.build_all
```

Downstream code loads data through `jm/data.py`: `load_works()`, `text(id)`, `emb(id)`, `load_pairs(kind, split)`,
`load_seeds()`, `load_develop_future()` and `load_credit()`. Schemas, counts, the cleaning audit
and known limitations are in [`data/README.md`](data/README.md).

| Path | Contents |
|---|---|
| `data/manifest/` | `works_manifest.csv` (IDs and metadata), drop log, cleaning audit, pair, credit and embedding stats |
| `evals/pairs/` | EVALUATE pairs: `cross_*`, `cite_*`, `cross_fullms_*` (train/val/test) |
| `develop/` | DEVELOP `seeds.jsonl` and `future_manifest.jsonl` |
| `credit/` | CREDIT `players.json`, `future_T.json`, `citation_edges.jsonl` |
| `scripts/s01..s08` | pipeline steps, orchestrated by `data/build_all.py` |

*(Work in progress; see `results/` for figures as they are produced.)*
