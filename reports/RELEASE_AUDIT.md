# Release audit against the master task

Audit status: **PROVISIONAL — external gates remain open**. Final runtime tests: 154 passed; before/after source digest identical. Passing local tests do not imply competition score or verified Linux-container behavior.

| Requirement | Evidence | State |
|---|---|---|
| Read entire master prompt; begin PHASE 0 | artifacts/research/MASTER_TASK.md; source history | Done |
| Obtain PDF, baseline, official requirements and datasets | artifacts/official; artifacts/data; data_manifest.json | Done |
| Resolve official CLI/env/cwd/input/output contract | CONTRACT.md; scripts/baseline_smoke.py | Done |
| Root solution.py; same code for different tasks | solution.py; clean-ZIP regression/multiclass verification | Done |
| Allowed dependencies only; exact main version pins | capability-matrix.json; AST archive audit; both official requirement files | Done for used subset |
| No pretrained weights, external training data or alternate API keys in competition archive | source-only archive manifest; dev OpenRouter client/key excluded | Done for competition runtime; separate development backend explicitly authorized by user |
| Task/target/ID/metric inference and conflicts | task.py; parametrized schema tests | Done for documented supported schema |
| Regression, binary and multiclass; original labels/probability order | 18 actual model API fits, all label/probability tests | Done |
| Multiple outputs / unknown task handling | explicit UnsupportedTask or documented metric fallback | Limited; no multi-target learner |
| All requested core metrics and direction | metrics.py; sklearn identity tests | Done; probability formats must be unambiguous |
| Rare/unseen categories, missingness, numeric strings, time/text/IP | 35 synthetic scenarios × 2 seeds; real preprocessing/model fits | Done for tested cases |
| Group/time/stratified/duplicate-safe immutable split | validation.py; legality tests; leakage-ablation.json | Done |
| Fold-fitted preprocessing/frequency/aggregation, no target encoding leak | features.py; training-fold tests; runtime recipes exclude target and ID | Done |
| Anytime initial CSV and atomic improvements | submit.py; checkpoint timing in summaries; fault tests | Done |
| Portfolio and local HPO with fixed objective | 6 families; public histories and source snapshots | Done |
| Monotonic budget and finalization reserve | budget.py; budget tests; scaled deadline simulation | Done locally |
| Worker timeout, RSS guard, process-tree termination, limited environment | budget.py; execution tests | Done locally |
| Linux address-space cap and Docker CPU/RAM quota | worker.py; no Docker installed | UNVERIFIED |
| Error diagnostics, quantile residuals, class P/R/F1/confusion, calibration, group error, drift and target-copy signals | judge.py; task.py; experiment registry; training-fold tests | Done for bounded supported diagnostics |
| Recipe schema, operation/package/parameter/type restrictions | recipes.py; adversarial tests | Done |
| Generate and execute actual Python feature expressions, real fit and judge | generated recipe.py files; compiled-expression integration test | Done with deterministic/mock proposals |
| Qwen official protocol, bounded calls/tokens, timeouts, malformed JSON and breaker | agent.py; fake-transport protocol tests; real-public context compaction under 14 KB | Implemented; live BLOCKED |
| Full client → JSON DSL → generated code → real fit → trusted judge → feedback | tests/test_execution.py; wire-loop-test.log | Done with fake transport; no network |
| Genuine Qwen hypothesis → execution → metric → accept/reject | no organizer credentials; user confirmed unavailable | BLOCKED |
| User-authorized live free OpenRouter hypothesis → generated code → real fit → metric → reject/keep | openrouter-live.json; two actual hypotheses and zero-cost usage receipts | Done for MiniMax development backend; both hypotheses rejected; not an organizer-Qwen pass |
| Live vs deterministic equal-budget ablation | impossible without official API settings | BLOCKED |
| Mock vs deterministic paired equal-budget runs | paired-ablation.json; immutable source snapshot | Done for prototype: 16 paired runs; live Qwen still blocked |
| Independent confirmation and bounded ensemble / class multipliers | orchestration.py; ensemble.py; public/paired histories | Implemented; acceptance only on measured confirmation |
| Full refit and reproducible recipe / artifact digest | final_recipe.json; summary.json; generated code and split records | Done |
| Validity under SIGTERM, actual fit kill -9, OOM guard, atomic-finalization kill | tests/test_execution.py | Done locally |
| No dataset-name conditions in runtime | package_submission.py static audit | Done |
| 15–30+ private-like tasks | 35 scenarios, 70 runs; EVALUATION.md | Done |
| A/B/C/E/G model/HPO/features/ensemble/confirmation ablations | measured histories; explicit caveats on causal attribution | Measured component trials and versioned campaign histories; live D blocked |
| F random vs legal group/time ablation | leakage-ablation.json | Done |
| 3h-budget simulation with time_scale | deadline-simulation.json | Done; real wall-clock budget 60s |
| Real 3h soak if feasible | not executed | Not claimed |
| ZIP excludes data/venv/secrets/weights; root solution.py | release-manifest.json; archive allowlist | Done |
| Extract into empty directory and execute Python 3.12 official CLI | zip-verification.json | Done on macOS |
| Strict Linux Docker clean extraction run | Docker unavailable | UNVERIFIED |
| SHA256 / size / file list / test counts / risks | release-manifest.json; final-release-audit.json; EVALUATION.md; README.md | Done |
| Team daily limit and actual platform submit | no platform access used | Not performed / not claimed |

No release claim should be upgraded to fully verified while live Qwen and exact Linux-container gates remain open. The master task explicitly permits an honestly labeled provisional ZIP when external resources are unavailable.

## Final campaign and last safety changes

Final campaign: regression RMSE 42.728359 (confirmation 40.565357), classification Macro F1 0.655730 (confirmation 0.652610); full refits of both two-model ensembles succeeded. These are local holdouts, not leaderboard scores.

The public campaign used frozen runtime `0cb87bffe7f686f5f93b14a2067f062a50edcf82d9e70ef57dcdb00ae9faf5ac`. Subsequent changes fixed empty-category profiling, bounded researcher context compaction, request counters, diagnostics and output-schema inference. Public model TaskSpec inputs remain identical after excluding the added diagnostic-only `suspicious_leakage_features` field (profile-fix-equivalence.json). This comparison does not establish byte identity of all diagnostics or repeat the campaign on the final revision. The final runtime is separately tied to its own 154-test digest, deadline simulation and clean-ZIP verification.

The user confirmed organizer API settings are not available. A fresh local check also found none of the three official environment variables or a project `.env`. Docker was absent both from PATH and the standard Docker Desktop/Homebrew CLI locations. These two external gates remain unverified; no organizer credentials, platform submission or Linux pass are implied.

The user subsequently authorized an OpenRouter development backend and browser
access to obtain its key. Its client lives in scripts/openrouter_client.py and is
injected only through scripts/run_openrouter.py. The fixed model is
minimax/minimax-m3.1-flash:free; both provider price ceilings are zero. A live
two-hypothesis run produced valid generated-code fits, independent rejection of
both worse candidates and a valid full-refit ensemble CSV. API-reported cost was
zero. This confirms a live alternative backend, not Qwen compatibility or a
quality improvement. No payment details were added. The local key file is
ignored and has mode 0600; it is absent from tracked sources, reports and ZIP.

Runtime changes after the 148-test release are limited to explicit research
client injection, backend provenance and a configurable output-token allowance
(official Qwen keeps its original 1200-token default). Model fitting, validation,
metrics, submission rules and the automatic official provider selection did not
change. The release artifacts are rebuilt and rechecked for the new source hash.
