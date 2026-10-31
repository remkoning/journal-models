# Technical appendix: methods overview
*Bullets only. Status as of 2026-10-31.*

**Scope: a proof of concept.** v0 uses published abstracts and in-context or rulebook journal-models on a base model. The full project fine-tunes journal-models on much more of each journal's record: full texts, figures, tables and code, review reports, and decisions including rejections. It evolves richer rules from that record. See the README.

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
- **Data in the repo:** the processed corpus with abstracts and the embeddings are committed (OpenAlex, CC0). `python -m data.build_all` re-pulls everything from OpenAlex.

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
- **In-batch leakage and its fix.** Early runs batched 10 pairs per call. Every pair in a journal-model's batch contained that journal's paper, so the model could infer the journal from what the batch shared. The anonymized, no-information condition scored up to 82% instead of 50%. All pairs results now use **one pair per call**, and the anonymized no-information condition serves as the leakage check (it lands near 50%). With the leak removed, named models score 92% with the name only and 95% with contrastive abstracts. The earlier batched gain on Org Sci vs. MS was an artifact. Journal hidden ("Journal X"): 50% with no abstracts (chance, ties counted as ½) → 93% with 100 own abstracts only (97 / 95 / 80 by journal pair).
- **Attribution matrix:**
  - Held-out recent papers (2024–26 test; 40 per journal, 37 for MS) are scored by all three models (§4).
  - Each model's scores are centered to remove leniency differences.
  - Each paper is assigned to the model that rates it highest.
  - Results: 72% correct with the name only → 78% contrastive (chance 33%).

## 4. B. Calibration: "is this a *J* paper?"
- Each model gives every held-out paper a score from 0 to 100 for "would we publish this?" plus a short reason.
- Batches of 15 abstracts mix all three journals, in shuffled order.
- **Metrics:**
  - AUC for own vs. other journals' papers. Name only → contrastive: Org Sci 0.74 → 0.85, SMJ 0.82 → 0.93, MS 0.88 → 0.92.
  - **Journal hidden ("Journal X"), the version shown in Fig B** (`run_scoring --modes incontext,rulebook --anonymous`):
    - Own abstracts only: AUC 0.82 / 0.92 / 0.91; attribution 75%; the 61–80 band is 60% / 76% / 89% own.
    - Rules only: AUC 0.77 / 0.74 / 0.82; attribution 67%.
    - A model built only from a journal's own papers calibrates about as well as the named models.
  - Within each score band, the share of papers truly from *J*. This rises monotonically: 0% at ≤20, and 61% (Org Sci), 79% (SMJ) and 88% (MS) at 61–80, against a 33% base rate.
- **Vertical vs. horizontal** (`scripts/vertical_horizontal.py`), following the firm × application matrix of Alidaee, Khanna & Koning (2026):
  - All 117 papers × 3 models are re-scored with a different batch order (seed 1), giving two replicates.
  - A two-way random-effects ANOVA with replication splits score variance into paper (shared, vertical), paper × journal (journal-specific, horizontal) and noise. Each model's mean (leniency) is removed first.
  - The *within-venue* version also removes model means within each publishing journal, so venue sorting cannot drive disagreement.
  - Results:

    | Mode | Test-retest r | Within venue: shared | Within venue: journal-specific | Cross-model r within venue |
    |---|---|---|---|---|
    | Name only | 0.91–0.92 | 23% | 77% | 0.05 to 0.38 |
    | Contrastive | 0.95 | 0% | 100% | −0.30 to 0.21 |
    | Rulebook | 0.97–0.98 | 0% | 100% | −0.24 to 0.20 |

  - Caveats: published papers only, so quality is compressed. The question asked is fit ("would *J* publish this?"), not quality.

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
- **Reliability** = split-half correlation of value, averaged over 200 random half-splits, then Spearman–Brown. *Correction:* an earlier version used one odd/even split, which overstated reliability (0.20 → 0.07 named; 0.36 → 0.24 anonymized).
- **Pilot (Org Sci, m = 150, contrast design):**
  - Named, reliability 0.07: Claude's prior about the journal swamps any one abstract.
  - Anonymized ("Journal X"), reliability 0.24.
  - Correlation with citations is null at this scale.
- **Diagnosis:**
  - A ridge "datamodel" regression on all inclusion indicators does *not* raise reliability (0.14–0.18), so interference from the other abstracts is not the problem.
  - Probe position within the batch explains 4–7% of residual score variance.
  - A per-run leniency shift explains ~21%, but it cancels in the own-minus-other value.
  - **Per-probe scoring noise dominates.**
- **v2 design** (`--design own --anonymous --k 10 --n-probes 48`):
  - Hidden journal, own abstracts only (no contrast set), so all of the model's taste comes from the abstracts that vary.
  - k = 10, so each abstract carries a larger share.
  - 48 probes (24 own, 24 other) in one fixed order, so runs differ only in the subset.
  - m = 600. Results are compared with the pilot at matched m.
- **v2 results, m = 600 per journal:**

  | | Org Sci | SMJ | MS |
  |---|---|---|---|
  | Reliability per paper (200 random splits) | 0.62 | −0.02 | 0.70 |
  | Reliability per line of work (10 clusters) | 0.77 | ≈ 0 | 0.73 |
  | Reliability at m = 150 / 300 | 0.28 / 0.35 | 0.13 / 0.15 | 0.24 / 0.45 |
  | Mean own − other score | 11 | 46 | 26 |
  | Per-run AUC, own vs. other probes | 0.74 | 0.999 | 0.93 |
  | ρ(value, log citations) | 0.13 | −0.16 | −0.02 |

- **Org Sci details:**
  - Reliability 0.63 per paper, and 0.77 for 10 "lines of work" (k-means clusters of the taste abstracts' embeddings, credit averaged within cluster). In the pilot, clustering did not help.
  - At matched m the design gain is small: 0.28 vs. 0.24 at m = 150, and 0.35 at m = 300. Most of the gain is from more runs.
  - The highest credit goes to abstracts that sharpen Org Sci's *distinctive* taste: leader emotions and creativity, group norms, knowledge work in teams.
  - The most negative credit goes to abstracts that look like SMJ or MS work: profit distributions, microfinance trade-offs, managerial risk-taking. Including them blurs the model's taste.
  - Value vs. log citations: ρ = 0.13 (p = 0.19). Value vs. year: ρ = −0.09.
- **MS:**
  - Top credit goes to economics-style mechanism and policy papers: funding public goods, noncompete enforceability, gender quotas, management practices.
  - The most negative credit goes to sociological status papers.
  - This matches MS's evolved rulebook.
- **SMJ (ceiling):**
  - Any 10 SMJ abstracts let the model separate SMJ probes from the others almost perfectly (per-run AUC 0.999; exactly 1.0 in 69% of runs).
  - No single abstract can add anything, so credit is ≈ 0 everywhere. The abstracts are substitutes, and a contribution duplicated many times earns little marginal credit.
  - The per-run gap still varies (sd 5.6), but that is scale noise, not composition.
  - Fix: harder probes (topic-matched pairs, as in exhibit A) or smaller k.
- **Status:** a working pipeline, and a separate option rather than a core exhibit. Fine-tuning-based influence functions are a Stage 2 alternative.

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
  - **Rulebook models in calibration and feedback (named):**
    - Attribution 69% (vs. 72% name only, 78% contrastive); AUC .77 / .77 / .87.
    - Feedback pushes are *more* differentiated than with contrastive abstracts (χ² = 109 vs. 68).
    - Revisions move toward their own journal 73 / 70 / 90% of the time (vs. 57 / 53 / 77%).
    - Push profiles correlate with the contrastive ones at 0.81 (Org Sci), 0.24 (SMJ) and 0.95 (MS).
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
  - `python -m scripts.vertical_horizontal`
  - `python -m scripts.run_credit --journal OrgSci [--anonymous]` (pilot)
  - `python -m scripts.run_credit --journal OrgSci --design own --anonymous --k 10 --n-probes 48 --m 600` (v2), or `scripts/credit_overnight.sh SMJ MgmtSci`
  - `python -m scripts.make_figures`

## 9. Limitations
- **Published papers only.** Without rejections we cannot separate editorial taste from author sorting. Review data in Stage 2 would.
- **Style.** House style is part of what the models learn. We treat framing as part of taste, but cannot fully separate it from substance.
- **Small Management Science sample.** Only 287 core papers, so its CIs are wide.
- **Model drift.** Results depend on one base model, which will change.
- **Credit:** reliable for Org Sci at 600 runs (0.63 per paper); influence on an in-context model is not yet the Stage 2 influence on a fine-tuned one.
