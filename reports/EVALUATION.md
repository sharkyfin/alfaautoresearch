# Evaluation evidence

Current runtime source SHA256: `8b90ee3fe1646082082c3ec0b1a46e72503d9876d9726d6bfead1a24a317d7f8`.

Automated suite: **148 passed**, 13 warnings in 36.88s.
Command: `.venv/bin/python -m pytest tests -q --tb=short`.
Warnings in the current suite are upstream Matplotlib/pyparsing deprecations. Historical failure logs are retained; all-tests.log is the current result.

## Public task measurements

These are internal train-label holdouts, not organizer baseline scores or leaderboard estimates. Hidden test labels are never read.

| Task / run | Seed | Split | Linear reference | Selected objective | Confirmation objective | Seconds | Peak worker MiB | Accepted |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| regression / public-reg-360 | 42 | random | 48.5693 | 43.4456 | 41.5832 | 112.4 | 863.6 | 3 |
| classification / public-cls-360 | 42 | duplicate_group | 0.415773 | 0.64805 | 0.636453 | 301.6 | 1036.3 | 5 |
| regression / public-campaign/regression_42 | 42 | random | 48.5693 | 42.8169 | 41.5076 | 322.0 | 859.4 | 4 |
| classification / public-campaign/classification_42 | 42 | duplicate_group | 0.415773 | 0.64805 | 0.636453 | 728.2 | 1038.9 | 5 |
| regression / public-campaign/regression_53 | 53 | random | 48.8056 | 41.1766 | 40.7203 | 703.5 | 860.0 | 5 |
| classification / public-campaign/classification_53 | 53 | duplicate_group | 0.420753 | 0.618575 | 0.638291 | 864.5 | 841.6 | 4 |
| regression / public-final/regression_42 | 42 | random | 48.5693 | 42.7284 | 40.5654 | 768.0 | 677.8 | 4 |
| classification / public-final/classification_42 | 42 | duplicate_group | 0.415773 | 0.65573 | 0.65261 | 810.6 | 788.6 | 4 |

The second split is independent for selection confirmation; its absolute score is not directly compared to the first split. The two 360s exploratory runs predate the frozen source campaign. Each run preserves generated code, actual predictions, recipe hashes, source hashes, fixed splits, diagnostics and final recipe.

## Synthetic private-like coverage

35 scenarios × 2 seeds: 66 valid outputs, 4 explicitly expected unsupported tasks, 0 unexpected failures.
Among 62 scored nondegenerate tasks: 59 gains, 1 losses, 2 ties versus a training-fold dummy. No raw RMSE/F1 averaging is used.

| Scenario | Seed | Validation | Metric | Dummy | Model | Utility gain | Seconds |
|---|---:|---|---|---:|---:|---:|---:|
| iid | 17 | random | rmse | 3.10895 | 1.64828 | 1.46067 | 0.723 |
| skew | 17 | random | rmse | 57.4748 | 38.3623 | 19.1124 | 0.506 |
| negative | 17 | random | rmse | 2.41077 | 1.72771 | 0.683051 | 0.502 |
| small_high_p | 17 | random | rmse | 2.92365 | 2.33365 | 0.590002 | 2.368 |
| missing | 17 | duplicate_group | rmse | 2.36654 | 2.08839 | 0.278149 | 0.494 |
| unseen | 17 | random | rmse | 3.10895 | 1.64828 | 1.46067 | 0.513 |
| high_cardinality | 17 | random | rmse | 3.10895 | 2.16081 | 0.948134 | 0.379 |
| locale | 17 | random | rmse | 3.10895 | 1.64828 | 1.46067 | 0.539 |
| future | 17 | time | rmse | 2.8529 | 1.48566 | 1.36724 | 0.822 |
| datetime_iid | 17 | random | rmse | 3.10895 | 1.6592 | 1.44975 | 0.853 |
| group | 17 | group | rmse | 2.39839 | 0.76462 | 1.63377 | 0.576 |
| duplicates | 17 | duplicate_group | rmse | 2.65159 | 2.37969 | 0.271902 | 0.476 |
| binary_imbalance | 17 | stratified | f1_macro | 0.435294 | 1 | 0.564706 | 0.484 |
| multiclass_imbalance | 17 | stratified | f1_macro | 0.285714 | 0.584795 | 0.299081 | 0.761 |
| singleton | 17 | stratified | f1_macro | 0.326087 | 0.326087 | 0 | 0.766 |
| integer_labels | 17 | stratified | f1_macro | 0.181818 | 0.719888 | 0.53807 | 0.802 |
| unicode | 17 | stratified | f1_macro | 0.181818 | 0.774207 | 0.592389 | 0.734 |
| shift | 17 | random | rmse | 3.10895 | 1.64828 | 1.46067 | 0.500 |
| noise | 17 | random | rmse | 3.10895 | 1.69504 | 1.41391 | 3.625 |
| text | 17 | random | rmse | 3.10895 | 2.15053 | 0.958411 | 0.616 |
| all_null | 17 | random | rmse | 3.10895 | 1.64828 | 1.46067 | 0.879 |
| ip | 17 | random | rmse | 3.10895 | 1.652 | 1.45694 | 0.771 |
| shuffled | 17 | random | rmse | 3.10895 | 1.64828 | 1.46067 | 0.711 |
| inf | 17 | random | rmse | 3.10895 | 1.99692 | 1.11202 | 0.718 |
| injection | 17 | random | rmse | 3.10895 | 1.64828 | 1.46067 | 0.753 |
| unknown_metric | 17 | random | rmse | 3.10895 | 1.64828 | 1.46067 | 0.846 |
| auc | 17 | stratified | roc_auc | 0.5 | 1 | 0.5 | 0.755 |
| logloss | 17 | stratified | logloss | 0.538267 | 0.0514923 | 0.486775 | 0.539 |
| average_precision | 17 | stratified | average_precision | 0.229167 | 1 | 0.770833 | 0.542 |
| mae | 17 | random | mae | 2.32723 | 0.904842 | 1.42239 | 0.635 |
| weighted_f1 | 17 | stratified | f1_weighted | 0.204545 | 0.789953 | 0.585408 | 0.819 |
| iid | 43 | random | rmse | 2.14077 | 0.875962 | 1.26481 | 0.534 |
| skew | 43 | random | rmse | 30.58 | 21.9733 | 8.60669 | 0.531 |
| negative | 43 | random | rmse | 1.39521 | 0.772249 | 0.622965 | 0.529 |
| small_high_p | 43 | random | rmse | 2.65682 | 2.68601 | -0.0291867 | 2.625 |
| missing | 43 | duplicate_group | rmse | 2.07529 | 1.85356 | 0.221733 | 0.509 |
| unseen | 43 | random | rmse | 2.14077 | 0.875962 | 1.26481 | 0.527 |
| high_cardinality | 43 | random | rmse | 2.14077 | 1.39748 | 0.743287 | 0.392 |
| locale | 43 | random | rmse | 2.14077 | 0.875962 | 1.26481 | 0.559 |
| future | 43 | time | rmse | 2.36556 | 1.09134 | 1.27423 | 0.886 |
| datetime_iid | 43 | random | rmse | 2.14077 | 0.939792 | 1.20098 | 0.890 |
| group | 43 | group | rmse | 2.6993 | 1.11636 | 1.58294 | 0.576 |
| duplicates | 43 | duplicate_group | rmse | 2.19548 | 1.26683 | 0.928653 | 0.530 |
| binary_imbalance | 43 | stratified | f1_macro | 0.44186 | 0.847619 | 0.405759 | 0.549 |
| multiclass_imbalance | 43 | stratified | f1_macro | 0.285714 | 0.742666 | 0.456951 | 0.806 |
| singleton | 43 | stratified | f1_macro | 0.326087 | 0.326087 | 0 | 0.797 |
| integer_labels | 43 | stratified | f1_macro | 0.189055 | 0.763444 | 0.574389 | 0.825 |
| unicode | 43 | stratified | f1_macro | 0.189055 | 0.765493 | 0.576438 | 0.788 |
| shift | 43 | random | rmse | 2.14077 | 0.875962 | 1.26481 | 0.556 |
| noise | 43 | random | rmse | 2.14077 | 1.08469 | 1.05608 | 3.768 |
| text | 43 | random | rmse | 2.14077 | 1.4157 | 0.725068 | 0.651 |
| all_null | 43 | random | rmse | 2.14077 | 0.875962 | 1.26481 | 0.860 |
| ip | 43 | random | rmse | 2.14077 | 0.845771 | 1.295 | 0.769 |
| shuffled | 43 | random | rmse | 2.14077 | 0.875962 | 1.26481 | 0.742 |
| inf | 43 | random | rmse | 2.14077 | 1.15224 | 0.988531 | 0.684 |
| injection | 43 | random | rmse | 2.14077 | 0.875962 | 1.26481 | 0.732 |
| unknown_metric | 43 | random | rmse | 2.14077 | 0.875962 | 1.26481 | 0.700 |
| auc | 43 | stratified | roc_auc | 0.5 | 0.963158 | 0.463158 | 0.740 |
| logloss | 43 | stratified | logloss | 0.511823 | 0.194881 | 0.316942 | 0.728 |
| average_precision | 43 | stratified | average_precision | 0.208333 | 0.879389 | 0.671056 | 0.807 |
| mae | 43 | random | mae | 1.66604 | 0.490776 | 1.17527 | 0.910 |
| weighted_f1 | 43 | stratified | f1_weighted | 0.224502 | 0.788392 | 0.563889 | 0.801 |

Ambiguous dictionary targets and unsupported multi-target samples are classified as unsupported; these cases are not counted as successful model runs. Constant-target and one-class tasks receive a valid constant prediction.

## Equal-budget ablation

**Live Qwen ablation is BLOCKED:** user confirmed organizer credentials/endpoint/model settings are not available. Zero live calls were made. The following is a mock-infrastructure comparison and cannot establish Qwen research value.

| Task | Seed | Deterministic | Mock agent | Utility difference | Deterministic seconds | Mock seconds |
|---|---:|---:|---:|---:|---:|---:|
| iid | 17 | 0.537237 | 0.536084 | 0.00115266 | 32.72 | 31.09 |
| skew | 17 | 10.3535 | 9.5066 | 0.846893 | 29.63 | 31.33 |
| integer_labels | 17 | 0.900164 | 0.900164 | 0 | 40.69 | 37.85 |
| multiclass_imbalance | 17 | 0.843269 | 0.835371 | -0.00789849 | 31.57 | 31.17 |
| iid | 43 | 0.342654 | 0.346005 | -0.00335107 | 31.12 | 32.88 |
| skew | 43 | 10.8688 | 12.6203 | -1.75152 | 31.10 | 38.14 |
| integer_labels | 43 | 0.827451 | 0.830569 | 0.00311753 | 33.41 | 34.02 |
| multiclass_imbalance | 43 | 0.939011 | 0.936499 | -0.00251213 | 31.14 | 30.15 |

Each finalized pair uses the prototype runtime SHA256 recorded in paired-ablation.json. Later safety fixes and portfolio additions are tested separately. Each pair uses the same data, seed, budget and configured CPU threads. Source is frozen in runs/paired/source_snapshot. Earlier mixed-revision runs are separately labeled exploratory and excluded from this table. Sample size is insufficient for statistical significance claims.

Across 8 prototype pairs: 3 wins, 4 losses, 1 ties; win rate 37.5%. Median normalized improvement -0.134%, lower quartile -0.947%. Normalization is signed objective difference divided by the absolute deterministic objective on each task; this is not the competition scoring formula.

## Leakage and failure tests

- group: random_unsafe, RMSE 2.99163.
- group: group, RMSE 5.6131.
- time: random_unsafe, RMSE 0.0178971.
- time: time, RMSE 3.11574.

Actual kill -9 during fit, SIGTERM orchestration, memory-limit termination, worker timeout, invalid output rejection and kill immediately before os.replace are covered by tests/test_execution.py and tests/test_core.py. The previous CSV remains valid.

Scaled 3h scheduling simulation: nominal 10800s × 1/180 = 60s actual budget; see deadline-simulation.json. This is **not a real 3-hour soak**.

## Release limits

- macOS development host: 12 logical CPUs, 16 GiB RAM; worker threads capped at 8.
- Docker is unavailable locally. Linux RLIMIT_AS path and exact 8-CPU/32-GiB Docker environment still require verification.
- Archive was unpacked into a clean temporary directory and the official root CLI was exercised for regression and multiclass, with SIGTERM after a fitted checkpoint.
- No platform submission has been made. Daily team submission allowance cannot be inferred locally.
- Pretrained models/external training datasets are absent from runtime and ZIP.

The next most valuable experiment is a live organizer-Qwen paired ablation on the frozen suite at equal time budgets; first verify its endpoint and JSON protocol. Until settings are available, the archive remains explicitly provisional.
