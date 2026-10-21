# Kaggle development benchmark

External development metrics; no hackathon leaderboard score is estimated.

| Dataset | Mode | Metric | External | Internal | Seconds | Qwen fits | Status |
|---|---|---|---:|---:|---:|---:|---|
| california_housing | lightgbm | rmse | 45699.6 | 47219 | 8.6 | 0 | ok |
| california_housing | deterministic | rmse | 45442.9 | 46780.8 | 34.7 | 0 | ok |
| california_housing | live-agent | rmse | 45442.9 | 46780.8 | 50.1 | 0 | ok |
| adult_income | lightgbm | f1_macro | 0.815002 | 0.814016 | 7.6 | 0 | ok |
| adult_income | deterministic | f1_macro | 0.818145 | 0.816124 | 31.2 | 0 | ok |
| adult_income | live-agent | f1_macro | 0.818145 | 0.816124 | 47.3 | 0 | ok |
| dry_beans | lightgbm | f1_macro | 0.942378 | 0.935912 | 18.3 | 0 | ok |
| dry_beans | deterministic | f1_macro | 0.946461 | 0.938223 | 76.3 | 0 | ok |
| dry_beans | live-agent | f1_macro | 0.946461 | 0.938223 | 82.4 | 0 | ok |

RMSE: lower is better. Macro F1: higher is better. Raw metrics are never averaged.

Source SHA256: `6778485712c29058303916a56480400352e6c38de6e4e29785026a6bc3157362`.
The same train/evaluation splits and time cap are used for all modes within a dataset/seed.
Qwen is not shown the external labels or the external metrics during its run.
The 20% final lockbox remains unused unless --final is explicitly selected.
These data and development scripts are excluded from the submitted solution ZIP.
