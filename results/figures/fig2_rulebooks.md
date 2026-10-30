# Fig 2. What each journal values: evolved rulebooks (selected rules, verbatim)

Each journal-model's rulebook was evolved by a self-improving loop. In each round, Claude judges topic-matched pairs ("which of these did *J* publish?"), reads its mistakes, rewrites at most 20 one-line rules, and keeps a revision only if it holds up on validation pairs. The rules below are quoted verbatim; we selected them to show contrasts. The full rulebooks and every changelog entry, with the evidence behind each edit, are in `results/harness/`.

| | **Organization Science** | **Strategic Management Journal** | **Management Science** (strategy/org/E&I) |
|---|---|---|---|
| **Values** | "Theorizes how audiences read signals, claims, categories or actor attributes to form evaluations, emphasizing social meaning-making rather than testing economic demand or discrimination models." | "The firm, or the founder acting for the venture, is the decision-maker, and the abstract studies a strategic choice such as entry, positioning, investment, scope or innovation direction." | "Economics-style mechanisms such as incentives, information asymmetry, signaling, matching, equilibrium, or explicit trade-offs, rather than legitimacy, identity, or institutional logics." |
| | "Posits countervailing effects, paradoxes or trade-offs, such as a benefit at one stage or condition and a penalty at another, and specifies contingencies." | "The outcomes are at firm level: performance, value creation, growth-option value, revenue or competitive advantage, not only individual or audience outcomes." | "Credible causal identification (policy shocks, sudden events, instruments, randomization) beats panel associations with moderators, even on large samples." |
| | "Theorizes interplay between formal controls (monitoring, incentives, contracts) and informal or normative ones (values, trust, relationships), including substitution or crowding out." | "Hypotheses come from core strategy theories (resource-based, real options, behavioral theory, capabilities, competitive dynamics, transaction costs) and are tested with archival, experimental or qualitative evidence." | "Reports effect sizes in interpretable units (percentages, per-unit responses, gap closed), not just support for hypotheses' direction." |
| **Red flags** | "Economics-style effect estimation, such as percentage-point gaps or policy evaluation using exogenous variation, closing with efficiency or policy prescriptions rather than organizational mechanism." | "The outcome is a personnel or personal reward (pay, turnover, promotion, executive compensation, gender gaps) with no measured effect on firm performance or strategic action." | "Inductive qualitative work: grounded theory, single in-depth case studies, ethnography, or process models built from interviews." |
| | "Explains career or productivity outcomes through individual skill or human-capital accumulation, without social, relational, cognitive or organizational process." | "A sociological focus on categories, identity or evaluator perception, framed mainly within organization theory, where the firm's strategic value is absent." | "Long verbal theorizing ('we/I theorize', inverted U, lists of moderators) where the data merely confirm predicted directions without identification." |

**One reading:** the same feature can be a value for one journal and a red flag for another.
- *Quantified causal effect estimates* are a core value for Management Science and a red flag for Organization Science.
- *Sociological constructs* (categories, identity, evaluator perception) are central for Organization Science and a red flag for SMJ, unless the firm's strategic value is at stake.

**How well the rules alone carry each journal's taste.** Topic-matched test pairs, journal name hidden, one pair per call:
- With nothing: **50%** (chance).
- With the ~20 rules only: **86%** [82–89].
- With 100 of the journal's abstracts: **93%**.

About 20 readable rules recover roughly 85% of the above-chance signal, and they beat the abstracts on the hardest pair (Org Sci vs. Management Science: 85% vs. 80%).
