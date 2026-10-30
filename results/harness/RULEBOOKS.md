# What each journal's board learned to value

Each rulebook was evolved from an empty start by a judge-reflect loop with hard-example mining on topic-matched abstract pairs, and kept only if it did at least as well as the best so far on held-out validation pairs judged one pair per call (design in [REPORT.md](REPORT.md)). **Rules and changelog entries are verbatim model output.** Sections marked *model-written summary* are a separate Claude pass over those texts. Paste-ready system prompts: `{J}/final.md`, or `harness.load.rulebook_system(J)`.

| Journal | Rulebook | Val accuracy, empty → rulebook | Val, OrgSci–MgmtSci pairs | Test, empty → rulebook |
|---|---|---|---|---|
| OrgSci | v3 | 80.0 → 88.3 | 60.0 → 76.7 | 91.8 → 89.1 (n = 147) |
| SMJ | v5 | 91.7 → 93.3 | — | 92.5 → 85.6 (n = 146) |
| MgmtSci | v4 | 96.7 → 96.7 | 93.3 → 93.3 | 98.5 → 95.5 (n = 67) |

## What each journal values

| # | Organization Science (v3) | Strategic Management Journal (v5) | Management Science (v4) |
|---|---|---|---|
| 1 | Theorizes how audiences read signals, claims, categories or actor attributes to form evaluations, emphasizing social meaning-making rather than testing economic demand or discrimination models. | The firm, or the founder acting for the venture, is the decision-maker, and the abstract studies a strategic choice such as entry, positioning, investment, scope or innovation direction. | Economics-style mechanisms such as incentives, information asymmetry, signaling, matching, equilibrium, or explicit trade-offs, rather than legitimacy, identity, or institutional logics. |
| 2 | Posits countervailing effects, paradoxes or trade-offs, such as a benefit at one stage or condition and a penalty at another, and specifies contingencies. | The outcomes are at firm level: performance, value creation, growth-option value, revenue or competitive advantage, not only individual or audience outcomes. | Reports effect sizes in interpretable units (percentages, per-unit responses, gap closed), not just support for hypotheses' direction. |
| 3 | Explains with sociological or behavioral constructs such as status, legitimacy, categories, identity, authenticity, cognition, tokenism or symbolic compliance, not mainly economic efficiency. | Effects depend on contingencies: the abstract shows how strategy–environment fit or firm and industry characteristics strengthen or weaken the main relationship. | Credible causal identification (policy shocks, sudden events, instruments, randomization) beats panel associations with moderators, even on large samples. |
| 4 | Links individual cognition or behavior to organizational structures, formal arrangements or group processes, e.g., how contract terms or hierarchy create pressures that shape decisions. | The abstract names the costs, trade-offs or unintended penalties of strategic moves for the firm, such as change, repositioning, path dependence or departing from tradition. | A formal model (real options, matching, game, optimal choice) that derives predictions, especially when tested by field experiment or data, even if magnitudes are unstated. |
| 5 | Advances core organization-theory conversations: organizational learning, exploration and exploitation, search, optimal distinctiveness, categories, entrepreneurial creation, status dynamics. | Individual-level mechanisms count when they explain strategic actions or firm results; lab, manager or in-firm field experiments that measure firm-level outcomes such as revenue qualify. | A new, validated, reusable measure that predicts real outcomes; not a critique or correction of an existing measure. |
| 6 | Frames the contribution as resolving a tension between prevailing theory and emerging evidence, or between competing mechanisms, not just estimating an effect. | Governance and leadership: boards, top teams, CEO succession or dismissal, and executive motives such as status rivalry that drive strategic actions and their value consequences. | Large-sample archival, administrative, platform, or field data on actual behavior over small samples, vignettes, or single cases. |
| 7 | Uses experiments, quasi-experiments or field data to separate mechanisms or reveal hidden trade-offs of a management practice, not only whether it works. | Hypotheses come from core strategy theories (resource-based, real options, behavioral theory, capabilities, competitive dynamics, transaction costs) and are tested with archival, experimental or qualitative evidence. | Shows how decision makers (firms, employers, evaluators, boards) actually respond to constraints, with concrete managerial or policy implications. |
| 8 | Accepts formal or simulation models of organizational decision processes, such as judgment aggregation, learning, search or unintended consequences of practices, deriving propositions rather than optimal incentive designs. | The abstract explicitly claims contributions to strategy literatures, such as competitive, corporate, complementor, entrepreneurial or international strategy, and states what managers should do. | Precise, decision-relevant formalization of fuzzy strategy concepts that produces tools usable in empirical work. |
| 9 | Builds moderators from named theoretical mechanisms about actors, audiences or organizations, rather than from convenient proxies or practical boundary conditions. | Customer or audience reactions are framed as determinants of the value or cost of a firm's strategic choice, not as phenomena of evaluator cognition in their own right. | When both abstracts are sociological, prefer the one testing behavioral outcomes on large archival data over qualitative or multimethod case work. |
| 10 | Theorizes interplay between formal controls (monitoring, incentives, contracts) and informal or normative ones (values, trust, relationships), including substitution or crowding out. | Reviews, conceptual syntheses, or replications and re-examinations of influential firm-level studies that guide strategy researchers fit, even without new theory or data. | Questions about bias, discrimination, or inequality framed as testable demand-side or incentive mechanisms rather than as contextualizing institutional forces. |
| 11 | Explains individual or team outcomes through intraorganizational relationships, networks, social integration or position in formal and informal hierarchies. | Network positions, political connections or stakeholder ties are treated as firm-level sources of advantage, resource access, risk or boundary choices, often under environmental shocks. | Tests a common management practice (rankings, pay, hiring tools) on firm records, ruling out alternative mechanisms or showing standard remedies fail. |
| 12 |  | Qualitative studies fit when they show how strategy-making processes (formulation, collaboration, participant selection) produce different strategic outcomes, linked to strategy research. | Studies innovation as search or problem-solving behavior measured on novel behavioral data, rather than audience perceptions of novelty or distinctiveness. |

## What each journal does not value / red flags

| # | Organization Science (v3) | Strategic Management Journal (v5) | Management Science (v4) |
|---|---|---|---|
| 1 | Economics-style effect estimation, such as percentage-point gaps or policy evaluation using exogenous variation, closing with efficiency or policy prescriptions rather than organizational mechanism. | The main contribution is a formal model deriving optimal policies, equilibria or new mathematical constructs, without empirical testing or a synthesis for strategy research. | Inductive qualitative work: grounded theory, single in-depth case studies, ethnography, or process models built from interviews. |
| 2 | Principal-agent, mechanism-design or relational-contract models where a principal optimizes incentives, matching or challenges; self-enforcement or discounted-value logic suggests economics journals. | Economics or finance framing: corporate-finance mechanisms, market-price tests, IO-style effect-size estimates without management theory, or macro and public-policy outcomes. | Contribution framed around named literatures or coined labels (a new 'paradox', a named hypothesis, optimal distinctiveness, categories) rather than a substantive answer. |
| 3 | The main contribution is a new measure, dataset or descriptive typology rather than a theoretical mechanism about organizations or audiences. | The outcome is a personnel or personal reward (pay, turnover, promotion, executive compensation, gender gaps) with no measured effect on firm performance or strategic action. | Long verbal theorizing ('we/I theorize', inverted U, lists of moderators) where the data merely confirm predicted directions without identification. |
| 4 | Framed for strategy audiences, such as competitive advantage, M&A or capability performance, or industry evolution, with thin behavioral mechanism; also literature reviews or research agendas. | A sociological focus on categories, identity or evaluator perception, framed mainly within organization theory, where the firm's strategic value is absent. | Formal models full of coined jargon constructs and long lists of propositions, without a crisp economic insight or empirical tie-in. |
| 5 | Main explanation is cross-national legal origins or country regulation with legal implications; broad social context like generalized trust is fine when theorized as shaping organizing. | Inductive case studies building process theory for organization-theory debates (meaning-making, change resistance, identity) rather than explaining strategic outcomes suggest another outlet. | Purely conceptual critique or redefinition of a construct with only verbal or illustrative arguments. |
| 6 | Openly exploratory pattern-finding whose mechanisms are guessed after the fact, with no theoretical argument set out in advance. | Conceptual or simulation arguments about organizational adaptation or reconciling paradigms that are not anchored in a core strategy theory. | Context-bound descriptive findings (one occupation or sector) framed as institutional or contextual nuance rather than a general mechanism. |
| 7 | Explains career or productivity outcomes through individual skill or human-capital accumulation, without social, relational, cognitive or organizational process. | Executive or political behavior explained mainly by individual motives or ideology (donations, self-promotion), rather than as firm strategy affecting firm resources or value. | Multimethod designs that combine a case narrative with hypothetical-vignette experiments to illustrate a perception mechanism. |
| 8 | Tests a classic economic theory, such as taste-based discrimination, network effects or foreclosure, against market outcomes like prices, revenues or sales. | An intervention or program is evaluated only on individual-level outcomes, without theory about strategic choices or firm-level results such as revenue or productivity. | Methodological critique of an existing measure pitched as guidance for 'organizational' or strategy scholars rather than answering a substantive question. |

*Automatic lint: these rules contain a quoted phrase. Most are generic stock phrases, but check that none echoes the wording of one specific training abstract:* MgmtSci: "Contribution framed around named literatures or coined labels (a new 'paradox', a named hypothesis, optimal distinctiveness, categories) rather than a substantive answer."; MgmtSci: "Long verbal theorizing ('we/I theorize', inverted U, lists of moderators) where the data merely confirm predicted directions without identification."; MgmtSci: "Methodological critique of an existing measure pitched as guidance for 'organizational' or strategy scholars rather than answering a substantive question."

## Differences across journals (model-written summary)

- OrgSci values social meaning-making: status, legitimacy, categories, identity. MgmtSci wants economic mechanisms like incentives, signaling and matching. SMJ cares about these only when they shape the value of a firm's strategy.
- Unit of analysis: SMJ wants the firm as decision-maker with firm-level outcomes. OrgSci links individuals to organizational structures and networks. MgmtSci studies how any decision maker responds to constraints.
- Evidence: MgmtSci rewards causal identification, effect sizes and large behavioral datasets, and flags inductive qualitative work. OrgSci and SMJ accept qualitative studies when they serve their own theory debates.
- Formal models: MgmtSci welcomes optimal-choice, game and matching models. OrgSci accepts decision-process simulations but flags principal-optimization models. SMJ flags formal models that lack empirical tests or a synthesis for strategy research.
- SMJ accepts reviews, replications and statements of what managers should do. OrgSci flags reviews and research agendas. MgmtSci flags conceptual critiques and contributions framed around named literatures.
- All three reward contingencies or trade-offs. OrgSci frames them as paradoxes, SMJ as strategy–environment fit, and MgmtSci as explicit economic trade-offs with stated magnitudes.

## What each board learned to drop or revise

### Organization Science

*Model-written summary:*

- Audience-evaluation studies that test economic discrimination or demand using market revenues no longer count as OrgSci; the rule now requires social meaning-making.
- Moved away from accepting formal models in general; principal-agent and incentive-optimization models now count against OrgSci, and only decision-process or learning models count for it.
- Weakened the flag on experiments and policy-style evaluations of organizational practices; experiments that reveal trade-offs of a practice are now valued.
- Dropped cross-national culture or institutions as a red flag in itself; regional generalized trust is fine when theorized as shaping organizing.

*Verbatim changelog entries from accepted revisions* (4 of 6 proposed revisions were rejected on val and reverted; their changes are not listed here but are in `OrgSci/changelog_v*.md`):

- v3 **edited**: Theorizes how audiences such as investors, analysts, customers or job seekers read signals, claims, categories or actor attributes to form evaluations.
  - *became*: Theorizes how audiences read signals, claims, categories or actor attributes to form evaluations, emphasizing social meaning-making rather than testing economic demand or discrimination models.
  - *why*: The rule wrongly favored consumer-response tests of economic discrimination theory based on market revenues over within-organization network studies.
- v3 **edited**: Links individual cognition or behavior to organizational systems, structures, governance arrangements or group processes, explaining how context changes what individuals do.
  - *became*: Links individual cognition or behavior to organizational structures, formal arrangements or group processes, e.g., how contract terms or hierarchy create pressures that shape decisions.
  - *why*: OS published executive-contract work explaining decisions through felt pressures; the board did not credit this behavioral link.
- v3 **edited**: Uses quasi-experiments or field data to tell theoretical mechanisms apart (e.g., symbolic versus substantive), not only to test whether an intervention works.
  - *became*: Uses experiments, quasi-experiments or field data to separate mechanisms or reveal hidden trade-offs of a management practice, not only whether it works.
  - *why*: OS experiments on management practices that reveal trade-offs or interacting effects were underweighted against the alternatives.
- v3 **edited**: Accepts formal, computational or simulation models that derive propositions about actors' beliefs, misalignment, learning or organizational processes.
  - *became*: Accepts formal or simulation models of organizational decision processes, such as judgment aggregation, learning, search or unintended consequences of practices, deriving propositions rather than optimal incentive designs.
  - *why*: OS formal models study decision aggregation or unintended consequences; principal-optimization models went to other outlets, and the old rule did not tell them apart.
- v3 **edited**: Economics-style evaluation of a program, law or practice using exogenous variation, closing with efficiency or policy prescriptions instead of organizational theory.
  - *became*: Economics-style effect estimation, such as percentage-point gaps or policy evaluation using exogenous variation, closing with efficiency or policy prescriptions rather than organizational mechanism.
  - *why*: The non-OS articles reported percentage-point gaps; the old wording also penalized OS experiments on organizational practices.
- v3 **edited**: Formal incentive or contracting models centered on leverage, wages or effort efficiency, backed by stylized correlations, suggest economics or finance journals.
  - *became*: Principal-agent, mechanism-design or relational-contract models where a principal optimizes incentives, matching or challenges; self-enforcement or discounted-value logic suggests economics journals.
  - *why*: Non-OS articles used principal-agent optimization, matching mechanisms and self-enforcing contract logic; the board needed a sharper cue.
- v3 **edited**: Framed mainly around competitive advantage, industry evolution or strategic investment outcomes for strategy audiences, with thin sociological or behavioral mechanism.
  - *became*: Framed for strategy audiences, such as competitive advantage, M&A or capability performance, or industry evolution, with thin behavioral mechanism; also literature reviews or research agendas.
  - *why*: A counterpart framed around M&A capability microfoundations for corporate strategy readers, and a review anchor came from a strategy outlet.
- v3 **edited**: The main explanation is cross-national culture, legal origins or country institutions, with legal or regulatory implications, which points to international business or governance outlets.
  - *became*: Main explanation is cross-national legal origins or country regulation with legal implications; broad social context like generalized trust is fine when theorized as shaping organizing.
  - *why*: An OS article explained firm sourcing with regional generalized trust, and the old flag nearly penalized it.
- v3 **edited**: Explains outcomes through individual skill or human-capital advantages from learning-by-doing, without social, cognitive or organizational process.
  - *became*: Explains career or productivity outcomes through individual skill or human-capital accumulation, without social, relational, cognitive or organizational process.
  - *why*: A non-OS mobility study explained career gains through skill development, while the OS counterpart explained outcomes through relationships.

### Strategic Management Journal

*Model-written summary:*

- Dropped the requirement for archival or field data; lab experiments and qualitative evidence now qualify.
- Stopped treating conceptual work as a red flag in itself; conceptual extensions of core strategy theories, and replications without new theory, now fit.
- Inductive case studies are no longer penalized by default; the flag now applies only when the contribution is aimed at organization-theory debates rather than strategic outcomes.
- Reversed a hardened stance on personnel outcomes: retention, CEO departure and individual-level RCTs are acceptable when firm revenue or strategic action is measured.

*Verbatim changelog entries from accepted revisions* (3 of 6 proposed revisions were rejected on val and reverted; their changes are not listed here but are in `SMJ/changelog_v*.md`):

- v4 **edited**: Individual or employee-level mechanisms (micro-foundations) are explicitly tied to firm-level results, such as leaders affecting performance or worker interventions raising revenue.
  - *became*: Individual-level mechanisms count when they explain strategic processes (search, competitive response, decision-making) or firm results; lab or manager experiments qualify.
  - *why*: Lab experiments on organizational search and manager experiments on competitive response were SMJ, while an employee intervention study on turnover was not.
- v4 **edited**: Hypotheses are derived from core strategy theories (resource-based, real options, behavioral theory of the firm, capabilities, competitive dynamics) and tested on archival or field-experimental data.
  - *became*: Hypotheses come from core strategy theories (resource-based, real options, behavioral theory, capabilities, competitive dynamics, transaction costs) and are tested with archival, experimental or qualitative evidence.
  - *why*: SMJ articles used lab experiments and qualitative evidence, so a rule limited to archival or field data misled the board.
- v4 **edited**: A formal or analytical model deriving optimal design, pricing or fee policies without empirical testing suggests an operations, economics or marketing journal.
  - *became*: The main contribution is a formal model deriving optimal policies, equilibria or new mathematical constructs, without empirical testing or a synthesis for strategy research.
  - *why*: Formal papers that build new constructs went elsewhere, while SMJ took reviews of the same formal approach; the rule needs to separate the two.
- v4 **edited**: An economics or finance framing: testing information asymmetry with market-price predictions, macro or local economic outcomes, or impacts at the level of public policy.
  - *became*: Economics or finance framing: corporate-finance mechanisms, market-price tests, IO-style effect-size estimates without management theory, or macro and public-policy outcomes.
  - *why*: An IO-style entry-deterrence test and an internal capital markets paper went elsewhere, while their theory-driven strategy counterparts were SMJ.
- v4 **edited**: Personnel or labor outcomes such as exit, promotion, aspirations or gender gaps, studied without any link to firm strategy or firm performance.
  - *became*: The dependent variable is a personnel outcome such as turnover, exit, promotion, aspirations or gender gaps, even when it is framed as relevant to the firm.
  - *why*: A turnover study framed as relevant to the firm still went elsewhere, so framing it that way does not rescue an HR outcome.
- v4 **edited**: A sociological focus on categories, identity or evaluator perception where the producer is an individual and the firm's strategic value is absent.
  - *became*: A sociological focus on categories, identity or evaluator perception, framed mainly within organization theory, where the firm's strategic value is absent.
  - *why*: A diversification-authenticity study positioned in organization theory went elsewhere even though the producer was a firm.
- v4 **edited**: An inductive comparative case study that builds a process theory from a handful of firms suggests a qualitative organization-theory outlet.
  - *became*: An inductive case study that builds process theory framed in organization theory, with no tie to strategy-making or strategy literatures, suggests another outlet.
  - *why*: SMJ published inductive work on strategy-makers, so the flag should apply only when the strategy link is missing.
- v4 **edited**: Purely conceptual or simulation arguments about organizational adaptation, or about reconciling theoretical paradigms, with no empirical test of strategic outcomes.
  - *became*: Conceptual or simulation arguments about organizational adaptation or reconciling paradigms that are not anchored in a core strategy theory.
  - *why*: A conceptual extension of the resource-based view was SMJ, so being conceptual is not a red flag in itself.
- v5 **edited**: Individual-level mechanisms count when they explain strategic processes (search, competitive response, decision-making) or firm results; lab or manager experiments qualify.
  - *became*: Individual-level mechanisms count when they explain strategic actions or firm results; lab, manager or in-firm field experiments that measure firm-level outcomes such as revenue qualify.
  - *why*: SMJ published in-firm field experiments on employees when they also measured firm revenue; the board was penalizing them.
- v5 **edited**: The study covers strategic leadership and governance: top teams, dominant coalitions, family control or CEO succession, and their link to firm strategy or performance feedback.
  - *became*: Governance and leadership: boards, top teams, CEO succession or dismissal, and executive motives such as status rivalry that drive strategic actions and their value consequences.
  - *why*: SMJ articles covered board dismissal decisions and CEO motives that drive acquisitions; the rival articles covered CEO personal rewards.
- v5 **edited**: Reviews, essays or conceptual pieces that synthesize, extend or guide use of a strategy theory for strategy researchers fit, even without new data.
  - *became*: Reviews, conceptual syntheses, or replications and re-examinations of influential firm-level studies that guide strategy researchers fit, even without new theory or data.
  - *why*: SMJ published a replication of an influential firm-survival study, and the board ranked it below a study with new theory.
- v5 **edited**: Network positions, prior relationships or stakeholder ties are treated as firm-level sources of advantage, risk or boundary choices, often under environmental shocks.
  - *became*: Network positions, political connections or stakeholder ties are treated as firm-level sources of advantage, resource access, risk or boundary choices, often under environmental shocks.
  - *why*: SMJ framed corporate political connections as firm access to public resources, while the rival focused on individual donor motives.
- v5 **edited**: Qualitative studies fit when they examine how strategy-makers formulate, justify or communicate strategy and explicitly link findings to strategy research.
  - *became*: Qualitative studies fit when they show how strategy-making processes (formulation, collaboration, participant selection) produce different strategic outcomes, linked to strategy research.
  - *why*: The SMJ qualitative study explained how strategy-process dynamics lead to different outcomes; the rival added to the literature on meaning-making and change.
- v5 **edited**: The dependent variable is a personnel outcome such as turnover, exit, promotion, aspirations or gender gaps, even when it is framed as relevant to the firm.
  - *became*: The outcome is a personnel or personal reward (pay, turnover, promotion, executive compensation, gender gaps) with no measured effect on firm performance or strategic action.
  - *why*: The old rule wrongly penalized SMJ papers that measured retention together with firm revenue, and CEO-departure governance papers; it should target personnel or pay outcomes with no firm link.
- v5 **edited**: An inductive case study that builds process theory framed in organization theory, with no tie to strategy-making or strategy literatures, suggests another outlet.
  - *became*: Inductive case studies building process theory for organization-theory debates (meaning-making, change resistance, identity) rather than explaining strategic outcomes suggest another outlet.
  - *why*: Both qualitative rivals used the word 'strategic'; the real difference is whether the claimed contribution is to organization-theory debates or to strategic outcomes.
- v5 **dropped**: The main contribution is settling a debate between theories of the firm, rather than explaining a strategic choice and what follows from it.
  - *why*: None of these pairs relied on it, and the slot was needed for a more frequent pattern about individual executive motives.
- v5 **edited**: The abstract mainly describes an intervention or training design and its causal effect, without theory-derived hypotheses about strategic choices or firm outcomes.
  - *became*: An intervention or program is evaluated only on individual-level outcomes, without theory about strategic choices or firm-level results such as revenue or productivity.
  - *why*: SMJ accepts RCTs that report firm-level results, so the flag should apply only when outcomes stay at the individual level.

### Management Science

*Model-written summary:*

- Stopped favoring critiques or corrections of existing measures; only new, validated measures count, and methodological critiques became a red flag.
- Dropped the expectation that formal models report magnitudes or quantify trade-offs empirically; models that derive predictions now count even when magnitudes are unstated.

*Verbatim changelog entries from accepted revisions* (4 of 6 proposed revisions were rejected on val and reverted; their changes are not listed here but are in `MgmtSci/changelog_v*.md`):

- v4 **edited**: Credible causal identification: policy shocks, before/after events, counterfactual benchmarks, instruments, or field data isolating one mechanism.
  - *became*: Credible causal identification (policy shocks, sudden events, instruments, randomization) beats panel associations with moderators, even on large samples.
  - *why*: Exogenous-shock papers beat large-sample associational panels with moderators but the board was only weakly confident.
- v4 **edited**: A formal model that yields clear, checkable results, ideally paired with empirical measurement that quantifies its trade-offs.
  - *became*: A formal model (real options, matching, game, optimal choice) that derives predictions, especially when tested by field experiment or data, even if magnitudes are unstated.
  - *why*: Model-derived hypotheses tested empirically, and analytical models of optimal choice, were underweighted against archival rivals.
- v4 **edited**: A new, validated, reusable measurement method (e.g., text-based measures validated several ways) that predicts real outcomes.
  - *became*: A new, validated, reusable measure that predicts real outcomes; not a critique or correction of an existing measure.
  - *why*: The measurement rule wrongly favored a critique of an existing measure over a policy-shock study.
- v4 **edited**: Contribution framed mainly as 'implications for research on' named literatures (optimal distinctiveness, ambidexterity, categories) rather than a substantive answer.
  - *became*: Contribution framed around named literatures or coined labels (a new 'paradox', a named hypothesis, optimal distinctiveness, categories) rather than a substantive answer.
  - *why*: Coined paradoxes and named hypotheses that extend sociological constructs marked the non-MS papers.
- v4 **edited**: Hypotheses stated as an inverted U or a list of moderators, with no effect magnitudes or identification strategy.
  - *became*: Long verbal theorizing ('we/I theorize', inverted U, lists of moderators) where the data merely confirm predicted directions without identification.
  - *why*: Non-MS papers, even experimental ones, devoted most of the abstract to theorized chains of predictions.
