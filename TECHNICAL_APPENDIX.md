# Technical appendix: methods overview
*Bullets only. Status as of 2026-10-30. Numbers marked [PROVISIONAL] are being re-run.*

## 1. Data
- **Source:** OpenAlex. Articles 2016–2026 from:
  - *Organization Science* (912)
  - *Strategic Management Journal* (1,014)
  - *Management Science* (287 in the strategy, organizations, and entrepreneurship & innovation departments; identified from the "accepted by …, \<department\>" line before it is stripped)
- **Cleaning:**
  - Strip venue tells: editor lines, SMJ section headers (only the Research Summary is kept), INFORMS boilerplate, copyright lines.
  - Mask journal self-references.
  - Truncate at a sentence boundary to ≤180 words.
  - Audit: residual giveaways are ~0%.
- **Splits:** random by paper (hash of OpenAlex ID): 50% train, 20% val, 30% test. Taste sets come only from train; every probe and pair comes from test.
- **Topic-matched pairs:** for each paper in journal *J*, the nearest neighbour in journal *K*. Conditions:
  - same OpenAlex subfield, or cosine ≥ 0.85;
  - cosine floor 0.75;
  - publication years within ±1;
  - greedy one-to-one matching.

  Embeddings: BAAI/bge-base-en-v1.5, run locally. Pairs share a subfield 96–98% of the time.
- **Copyright:** no abstract text is committed. `python -m data.build_all` rebuilds the local store from OpenAlex.

## 2. Journal-models
- **Base model:** Claude Opus 5.5, effort `low`, called headless through Claude Code (`jm/llm.py`). The API is an alternative backend.
- **Forms of taste:**

  | Form | System prompt |
  |---|---|
  | **plain** | "You are the editorial board of *J*…" (the name only) |
  | **incontext** | plus 100 of *J*'s train-split abstracts, stratified by year |
  | **contrast** (v0 default) | plus 100 "published elsewhere" abstracts from the other two journals |
  | **rulebook** | readable rules evolved by the self-improving harness (§7), with no abstracts |

- **Anonymized variant:** the journal is called "Journal X," so its taste can come only from the abstracts.
- **Reproducibility:** every call is cached by the hash of its full request and logged to `results/ledger.jsonl`.

## 3. A. Is taste learnable, and does it differ?
- **Pairs test:**
  - For each topic-matched test pair, each of the two relevant journal-models answers "which did *J* publish?" as p_A ∈ [0,1].
  - Each pair is shown in both orders.
  - Accuracy = P(own paper > 0.5), with bootstrap 95% CIs.
- **In-batch leakage and its fix.** Early runs batched 10 pairs per call. Every pair in a journal-model's batch contained that journal's paper, so the model could infer the journal from what the batch shared. The anonymized, no-information condition scored up to 82% instead of 50%. All pairs results now use **one pair per call**, and the anonymized no-information condition serves as the leakage check (it lands near 50%). With the leak removed, named models score 92% with the name only and 94% with contrastive abstracts. The earlier batched gain on Org Sci vs. MS was an artifact. Journal hidden ("Journal X"): 50% with no abstracts (chance, ties counted as ½) → 93% with 100 own abstracts only (97 / 95 / 80 by journal pair).
- **Attribution matrix:**
  - Held-out recent papers (2024–26 test; 40 per journal, 37 for MS) are scored by all three models (§4).
  - Each model's scores are centered to remove leniency differences.
  - Each paper is assigned to the model that rates it highest.
  - Results: 72% correct with the name only → 78% contrastive (chance 33%).

## 4. B. Calibration: "is this a *J* paper?"
- Each model gives every held-out paper a score from 0 to 100 for "would we publish this?" plus a short reason.
- Batches of 15 abstracts mix all three journals, in shuffled order.
- **Metrics:**
  - AUC for own vs. other journals' papers. Name only → contrastive: Org Sci 0.74 → 0.86, SMJ 0.82 → 0.93, MS 0.89 → 0.92.
  - Within each score band, the share of papers truly from *J*. This rises monotonically: 0% at ≤20, and 61% (Org Sci), 79% (SMJ) and 88% (MS) at 61–80, against a 33% base rate.

## 5. C. Same paper, three journals: feedback
- **Papers:** 30 held-out papers from 2025–26, 10 per journal. Each of the three contrastive journal-models returns:
  - a fit score;
  - what the journal values about the work;
  - three next-step suggestions;
  - a revised abstract.
- **Blind classification:** a classifier that never sees which journal wrote the suggestion tags all 270 suggestions into seven categories: theory, mechanism, methods/identification, data, scope, practice, positioning.
  - The journals' mixes differ: χ² = 68, df = 12, p < 10⁻⁹.
  - For 28 of 30 papers, the three journals' top category differs.
- **Robustness:**
  - A second classifier (Sonnet) agrees at κ = 0.73, with the same pattern (χ² = 62).
  - Exploratory split, chosen after seeing the data: separating "formal model" from verbal theory shows MS's theory push is almost entirely formal models (29% vs. 1% verbal); Org Sci's is verbal (34% vs. 1%).
- **Fit:** each model gives its own journal's papers the highest scores (51–56 vs. 28–38).
- **Revision direction:** we compare the embedding of each revised abstract with the centroids of each journal's train-split papers. The revision moves most toward the model's own journal 53–77% of the time (chance 33%).

## 6. D. Credit = influence on the journal-model
- **Design:**
  - m random subsets of k = 20 of *J*'s 100 taste abstracts. The contrast set is fixed, so only the own-journal subset varies.
  - In each run the model scores a fixed probe set: 24 recent test abstracts, 12 from *J* and 12 from the other journals.
  - β[i, p] = mean score of probe p when abstract i is included minus when it is excluded (a Banzhaf / datamodel estimator).
  - **Value** = β on own-journal probes minus β on other probes.
  - **Influence** = RMS of β, estimated from split halves.
- **Pilot (Org Sci, m = 150):**
  - With the journal named, split-half reliability is 0.20; Claude's prior about the journal swamps any one abstract.
  - Anonymized ("Journal X"), reliability is 0.36.
  - Correlation with citations is null at this scale.
  - Projection: about 600 runs per journal for reliability near 0.7.
- **Status:** a working pipeline that needs more runs. Fine-tuning-based influence functions are a Stage 2 alternative.

## 7. Self-improving rulebook harness
- A rulebook has two parts: ≤12 "what *J* values" rules and ≤8 "what *J* does not value" rules. It holds general principles only and never names a paper.
- **Loop, per journal:**
  1. Judge a pool of train pairs.
  2. Build a minibatch, mostly from errors and low-confidence cases.
  3. One reflection call (Opus, high effort) rewrites the rules and writes a changelog.
  4. Score on fixed val pairs, one pair per call.
  5. Accept if the score is ≥ the best − 1 point.
- **Output:** `results/harness/RULEBOOKS.md`, with the final rulebooks side by side, the rules dropped along the way, and the cross-journal differences. Changelogs record the evidence behind every edit.
- **Results (test pairs, one pair per call):**
  - Journal hidden, rules only: **86%** [82–89], vs. 50% with nothing and 93% with 100 abstracts. On Org Sci vs. MS: 85% vs. 80% for the abstracts.
  - Journal named, empty vs. final rulebook: 92 → 89 (Org Sci), 93 → 86 (SMJ), 99 → 96 (MS). Validation gains on 30 pairs did not generalize; the named prior already carries the taste.
  - Run: 6 iterations per journal, 3,549 CLI calls, 271 minutes (interrupted once by the plan's 5-hour limit and resumed from cache).

## 8. Infrastructure and cost
- **Backends:** Claude Code headless (on the author's plan, the default), the Anthropic API (capped) and OpenRouter (capped).
- **Billed API spend:** <$0.01. Notional plan usage is logged per call.
- **Reproduce:**
  - `python -m data.build_all`
  - `python -m scripts.pilot_taste --split test --batch 1 [--anonymous]`
  - `python -m scripts.run_scoring`
  - `python -m scripts.run_feedback`
  - `python -m scripts.feedback_robustness`
  - `python -m scripts.run_credit --journal OrgSci [--anonymous]`
  - `python -m scripts.make_figures`

## 9. Limitations
- **Published papers only.** Without rejections we cannot separate editorial taste from author sorting. Review data in Stage 2 would.
- **Style.** House style is part of what the models learn. We treat framing as part of taste, but cannot fully separate it from substance.
- **Small Management Science sample.** Only 287 core papers, so its CIs are wide.
- **Model drift.** Results depend on one base model, which will change.
- **Credit is underpowered** at pilot scale.
