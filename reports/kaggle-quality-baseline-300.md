# Kaggle development benchmark

External development metrics; no hackathon leaderboard score is estimated.

| Dataset | Mode | Metric | External | Internal | Seconds | Qwen fits | Status |
|---|---|---|---:|---:|---:|---:|---|
| california_housing | deterministic | rmse | 47076.3 | 44684.9 | 56.2 | 0 | ok |
| adult_income | deterministic | f1_macro | 0.813322 | 0.817145 | 41.1 | 0 | ok |
| dry_beans | deterministic | f1_macro | 0.93465 | 0.937017 | 40.0 | 0 | ok |

RMSE: lower is better. Macro F1: higher is better. Raw metrics are never averaged.

Source SHA256: `6778485712c29058303916a56480400352e6c38de6e4e29785026a6bc3157362`.
The same train/evaluation splits and time cap are used for all modes within a dataset/seed.
Qwen is not shown the external labels or the external metrics during its run.
The 20% final lockbox remains unused unless --final is explicitly selected.
These data and development scripts are excluded from the submitted solution ZIP.
