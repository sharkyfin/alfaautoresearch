# Kaggle development benchmark

External development metrics; no hackathon leaderboard score is estimated.

| Dataset | Mode | Metric | External | Internal | Seconds | Qwen fits | Status |
|---|---|---|---:|---:|---:|---:|---|
| california_housing | live-agent | rmse | 45603.7 | 43565.3 | 104.5 | 2 | ok |
| adult_income | live-agent | f1_macro | 0.813581 | 0.817457 | 45.2 | 2 | ok |
| dry_beans | live-agent | f1_macro | 0.933441 | 0.938585 | 61.2 | 0 | ok |

RMSE: lower is better. Macro F1: higher is better. Raw metrics are never averaged.

Source SHA256: `e35e12d7a73989e9e727b27f5f2f73e5b233d37b7b08317084e131b19a6f804d`.
The same train/evaluation splits and time cap are used for all modes within a dataset/seed.
Qwen is not shown the external labels or the external metrics during its run.
The 20% final lockbox remains unused unless --final is explicitly selected.
These data and development scripts are excluded from the submitted solution ZIP.
