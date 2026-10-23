**Frozen-source paired benchmark**

| Dataset | Seed | Arm | Mode | Metric | External | Seconds | Status |
|---|---:|---|---|---|---:|---:|---|
| concrete_strength | 42 | baseline | live-agent | rmse | 4.41582 | 84.8 | trained_success |
| concrete_strength | 42 | candidate | live-agent | rmse | 4.57739 | 77.1 | trained_success |
| dry_beans | 42 | baseline | live-agent | f1_macro | 0.949542 | 198.4 | trained_success |
| dry_beans | 42 | candidate | live-agent | f1_macro | 0.948477 | 182.8 | trained_success |

Stage: development. Evaluation labels remain in the independent parent process.
Original outer indices and source metrics are preserved; explicit metric-study overrides are recorded.
Group-labelled tasks report row scores and equally weighted group scores.
Unavailable/unrun cells stay empty with a reason. These offline metrics are not Boosters scores.

Summary: {"win": 0, "tie": 0, "loss": 2, "failure_or_unavailable": 0}
