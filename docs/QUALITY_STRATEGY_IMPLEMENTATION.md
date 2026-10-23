# Bounded quality strategy experiment (2026-10-23)

This is a separate experimental branch, `research/automl-challenger-strategy-20261023`. Its parent commit `91777bb` preserves the current engineering baseline (runtime `f85b2482…`). The original working tree and all protected ZIPs remain unchanged. No Boosters submission is authorized or performed.

## Confirmed limitations and changes

1. The normal search supports six model families. The already-installed official AutoGluon dependency was used only by a development reference script; it could not participate in the agent's independent confirmation or final refit. The new `atlas/challenger.py` is a bounded adapter through the existing Executor/worker/Judge, outside SearchController and AdaptiveTPE.
2. Inspection of AutoGluon 1.3.1's `DefaultLearner.general_data_processing` shows that a fresh feature generator normally learns on concatenated training and tuning rows. The adapter prefits its generator on training rows only. Bagging, stacking, dynamic stacking, calibration, GPU and neural models are disabled. Public API details: [fit](https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.fit.html).
3. Re-running an AutoML search on controls would select another model/ensemble on those controls. The adapter exports the selected leaf hyperparameters, trained iteration counts and ensemble weights from the primary split; controls fit those components without a new ensemble search. These are correctness constraints, not claims of a measured quality gain.
4. Existing Qwen could return only a concrete recipe, while deterministic research steps used a fixed feature order. Within the same existing API call, it can now choose one of seven eligible directions, optionally as a direction-only action. The controller compiles that action to the existing recipe DSL. Family reliability, costs, positive/negative history, champion diagnostics and remaining budget are provided; metric and validation remain immutable. A malformed or unavailable action falls back to the existing deterministic proposal.

## Integration and cost

AutoGluon starts only after a trained main champion. Its phase reserves at most `min(1800 seconds, 30% of total budget, available slack after main finalization)`, including two controls and refit; an individual search/control worker has a 360 second ceiling. Less than 20 seconds per worker skips the challenger. On a 10500 second task the total reserve is capped at 1800 seconds; it is drawn from the same deadline, not added to it. Refit gets a separately bounded share of this reservation and still runs under the existing watchdog and memory guard. Actual consumption will be recorded in the bounded benchmark report.

Supported validation policies: random, stratified, entity groups and exact-duplicate groups. Each worker verifies disjoint row coverage and group isolation. Time validation is deliberately skipped because an AG refit policy covering temporal gaps has not been verified. Missing global classes, unsupported datasets, absent/wrong AG version and insufficient budgets skip safely. Text-heavy tasks are left to the existing sparse/text search. Only classical GBM, CatBoost, RF and XT from task-local data are considered.

The challenger never publishes its primary-split prediction. It is added to the same frozen control procedure, robust selection, optional existing blend and atomic all-components refit. If training, policy export, confirmation or refit fails, the last good main/confirmed checkpoint remains. The custom AG scorer delegates to the exact canonical Metric implementation, including global Macro F1 class support and binary positive-class semantics. Refit reuses the primary artifact and AG's [refit_full](https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.refit_full.html), after checking that its cached train+tuning indices cover every known label.

Strategic directions: improve family, another family, semantic features, imbalance, target transform, complementary-family candidate, compute allocation. Only task-eligible directions/families are exposed; two failures quarantine a family. Effort quick/standard/deep adjusts one worker's cap using measured costs. Direction-only model/budget actions reuse initial recipes or AdaptiveTPE. Semantic directions must supply concrete controlled DSL features; no invented default column pairs are added. Complementarity remains a hypothesis until the existing independent ensemble checks confirm it.

## Explicitly rejected scope

No new threshold optimizer, OOF store, ensemble algorithm, TPE implementation or LLM layer. Existing threshold/OOF/blend mechanisms already cover those needs. No concrete measured Macro F1 gap has justified changing their selection behavior here. Stacking would require further independent evidence and budget beyond this bounded stage, so it is not added. No mass benchmarks or automatic benchmark continuation.

## Validation plan and limitations

58 targeted tests pass. Mocked adapter/orchestration tests cover failures in search, controls and refit, fixed policy propagation, probability columns, exact scorer alignment, unsupported validation, duplicate isolation, allowed strategies and API limits. Existing component compatibility checks are kept separate from the four full benchmark runs.

Exactly four full agent runs are declared in `reports/quality_strategy_20261023/comparison-plan.json`: before/after, Concrete Strength regression and Dry Beans classification, seed 42, 360 seconds, 8 CPUs, 8 GiB local guard and at most two organizer Qwen calls. Same immutable source archives, outer labels isolated in evaluator memory, identical input fingerprints and scorer. These development cases have been used before; they are not a new private/generalization test. Qwen is stochastic, so any observed difference measures the combined system rather than an isolated causal effect of either feature.

Remaining risks: limited two-case sample; no full three-hour organizer-image run; CPU/RAM trade-off may reduce normal search; the frozen prototype displaced the third main finalist; this was corrected in the final artifact to preserve all three main finalists and use a separately funded fourth slot; AG's pinned-version private ensemble-weight accessor must fail closed on unsupported topology; controls still use normal model early stopping as in the existing implementation. The previous best platform result remains the historical `submission_recovery_1258.zip` (1258.773); no new platform score is claimed.

## Post-freeze audit corrections

The four declared runs remain on the original frozen prototype `4588cb3f…` / ZIP `7644890c…`; they are not rerun or rewritten. Four corrections are applied only to the separately named final experimental artifact:

- AutoGluon RF/XT defaults include `n_jobs=-1` and `_fit` ignores `num_cpus` when that parameter is already present. On this 12-thread host the prototype therefore did not prove a strict 8-CPU cap for those models. Final code binds RF/XT `n_jobs`, GBM `num_threads` and CatBoost `thread_count` explicitly and forces CPU backends, with unit assertions for 2/8/12 requested CPUs.
- The prototype reserved time for three main finalists plus challenger but replaced the third main finalist in confirmation. The final code preserves all three original slots and adds challenger as the fourth candidate, using its existing separately reserved cost. A mocked orchestration test confirms all four families reach the controls. This preserves existing model diversity rather than discarding it to make room.

- The prototype classification challenger won frozen control selection, but refit failed before training because the local AG environment lacked `setuptools==80.9.0`, explicitly present in official requirements. AG `load()` unconditionally imports `pkg_resources`. Installing that official dependency allowed read-only reload of the existing artifact; no model was retrained. Final challenger availability requires both pinned AG and the official reload dependency; a mocked refit test verifies complete label coverage, original policy and bounded CPU settings.
- The final action menu omits the duplicate family-cost payload; Qwen continues to use the existing `research_evidence` cost/history section. Detailed medians remain inside the deterministic controller.

No fifth full ML run is authorized in this stage. Therefore the numerical before/after table describes the frozen prototype, while the final corrected ZIP has unit/static verification only. No improved quality is attributed to the corrected artifact without a new authorized comparison.
