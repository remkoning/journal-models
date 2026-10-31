# Journal Models (prototype)

Prototype accompanying the Organization Science special-issue proposal *"The Journal as Model"* (Koning & Myers).

**Idea.** A journal is a horizontally differentiated organization of scientific taste. A *journal-model* is a base LLM given one journal's taste. The taste can be carried as the journal's own abstracts (contrastive in-context) or as a readable rulebook evolved by a self-improving harness. Journals in v0: *Organization Science*, *Strategic Management Journal*, and *Management Science* (strategy, organizations, entrepreneurship & innovation).

> **Status: proof of concept.** v0 is deliberately simple: three journals, published abstracts only, and an off-the-shelf model given each journal's taste in context or as a short evolved rulebook. Its job is to show that journal taste is learnable, differs across journals, and can be made legible. Where we're headed:
> - **Fine-tuning:** journal-models trained on each journal's corpus and decisions, not just prompted.
> - **Much more data from the journal and its articles:** full texts (introductions, theory, figures, tables, code), review reports, and editorial decisions, including rejections.
> - **Richer evolved rules:** self-improving rulebooks learned from that fuller record, not from abstracts alone.
> - **Journals as multi-agent organizations:** editor, reviewer, methods and literature agents with their own tools.
>
> The numbers here are first results from v0, not estimates of what a full journal-model can do.

**Four questions, one set of models** (figures in `results/figures/`):

| | Question | Script |
|---|---|---|
| A | Is journal taste learnable, and does it differ? | `scripts/pilot_taste.py --split test --batch 1 [--anonymous]` · `scripts/run_scoring.py` |
| B | Calibration: does a high score mean it really is that journal's paper? Do the models share a quality ranking (vertical) or not (horizontal)? | `scripts/run_scoring.py` · `scripts/vertical_horizontal.py` |
| C | Same paper, three journals: where does each push the research? | `scripts/run_feedback.py` · `scripts/feedback_robustness.py` |
| D | Credit: how much does each training abstract shape the model? | `scripts/run_credit.py` |
| + | Self-improving rulebooks: what each journal values, in plain language | `harness/` → `results/harness/RULEBOOKS.md` |

Methods, in bullets: `TECHNICAL_APPENDIX.md`. Data card: `data/README.md`.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m data.build_all          # optional: the processed corpus is committed; this re-pulls it from OpenAlex
```

All LLM access goes through `jm/llm.py`. It has three backends:
- **`cli`**: Claude Code headless (`claude -p`). This is the default.
- **`api`**: the Anthropic API. Set `ANTHROPIC_API_KEY`.
- **`openrouter`**: set `OPENROUTER_API_KEY`.

Every call is cached and logged to `results/ledger.jsonl`. Billed backends are capped by `JM_CAP_API_USD` and `JM_CAP_OPENROUTER_USD`.
