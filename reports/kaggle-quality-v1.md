# Kaggle development benchmark

External development metrics; no hackathon leaderboard score is estimated.

| Dataset | Mode | Metric | External | Internal | Seconds | Qwen fits | Status |
|---|---|---|---:|---:|---:|---:|---|
| california_housing | deterministic | rmse | 45442.9 | 46780.8 | 33.4 | 0 | ok |
| california_housing | live-agent | rmse | 44811.1 | 46444.8 | 53.2 | 2 | ok |
| adult_income | deterministic | f1_macro | 0.818145 | 0.816124 | 29.8 | 0 | ok |
| adult_income | live-agent | f1_macro | 0.818145 | 0.816124 | 42.4 | 2 | ok |
| dry_beans | deterministic | f1_macro | 0.942773 | 0.938184 | 75.4 | 0 | ok |
| dry_beans | live-agent | f1_macro | 0.947376 | 0.937907 | 76.4 | 2 | ok |

RMSE: lower is better. Macro F1: higher is better. Raw metrics are never averaged.

Source SHA256: `e05ea89b2d2ecf84c2b11d63cd107f75b3fc447662e5bac090e45d88ef2f03c7`.
The same train/evaluation splits and time cap are used for all modes within a dataset/seed.
Qwen is not shown the external labels or the external metrics during its run.
The 20% final lockbox remains unused unless --final is explicitly selected.
These data and development scripts are excluded from the submitted solution ZIP.
