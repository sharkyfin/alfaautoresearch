# Kaggle development benchmark

External development metrics; no hackathon leaderboard score is estimated.

| Dataset | Mode | Metric | External | Internal | Seconds | Qwen fits | Status |
|---|---|---|---:|---:|---:|---:|---|
| california_housing | live-agent | rmse | 46177.3 | 45435.9 | 119.0 | 2 | ok |
| adult_income | live-agent | f1_macro | 0.823763 | 0.818669 | 112.6 | 1 | ok |
| dry_beans | live-agent | f1_macro | 0.937985 | 0.935492 | 89.2 | 1 | ok |

RMSE: lower is better. Macro F1: higher is better. Raw metrics are never averaged.

Source SHA256: `e2f776a1bdeb35a2010a1cba434733fd6edf6d97566cd045a832c1ee27dbb7d9`.
The same train/evaluation splits and time cap are used for all modes within a dataset/seed.
Qwen is not shown the external labels or the external metrics during its run.
The 20% final lockbox remains unused unless --final is explicitly selected.
These data and development scripts are excluded from the submitted solution ZIP.
