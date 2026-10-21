# Kaggle development benchmark

External development metrics; no hackathon leaderboard score is estimated.

| Dataset | Mode | Metric | External | Internal | Seconds | Qwen fits | Status |
|---|---|---|---:|---:|---:|---:|---|
| california_housing | live-agent | rmse | 45394.7 | 43535.7 | 150.0 | 2 | ok |
| adult_income | live-agent | f1_macro | 0.813581 | 0.817457 | 44.3 | 2 | ok |
| dry_beans | live-agent | f1_macro | 0.934369 | 0.937658 | 48.1 | 1 | ok |

RMSE: lower is better. Macro F1: higher is better. Raw metrics are never averaged.

Source SHA256: `f2d104a73fac2a01c7c615fee1bcfb0fd18c0faa52a87579b6f1ecd2ca122eb7`.
The same train/evaluation splits and time cap are used for all modes within a dataset/seed.
Qwen is not shown the external labels or the external metrics during its run.
The 20% final lockbox remains unused unless --final is explicitly selected.
These data and development scripts are excluded from the submitted solution ZIP.
