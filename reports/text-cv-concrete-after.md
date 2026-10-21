# Full Kaggle pipeline run

Dataset: [elikplim/concrete-compressive-strength-data-set](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set).
Executed the flat upload as `python3 solution.py task.md`; no agent defaults were overridden.
Local execution uses the existing virtual environment, not a fresh platform Docker image.
The unseen external split was fixed before launch; labels stayed outside the task folder.
Exact duplicate feature groups cannot cross the external train/test boundary.

Status: **completed**. Exit code: 0. Wall time: 70.705 seconds.
Training rows: 820; external test rows: 210; original features: 8.
Agent budget: 900.0 seconds; experiments: 33; Qwen requests: 3.
Completed Qwen trials: 3; primary-holdout accepted Qwen trials: 0; full refit: True.

External rmse: **3.755457**.
MAE: 2.706258; R²: 0.944380; training-mean baseline RMSE: 15.926242.

TPE: 12 completed trials; family proposals: {'hist': 8, 'lightgbm': 4}.
Frozen CV: 3 planned folds; 3 fully checked candidates.

| Candidate | Model | Fold scores | Mean | Standard deviation |
|---|---|---|---:|---:|
| exp_0004 | hist | 5.026616, 5.652592, 4.612416 | 5.097208 | 0.427574 |
| exp_0011 | hist | 5.026616, 5.652592, 4.612416 | 5.097208 | 0.427574 |
| exp_0026 | lightgbm | 5.072756, 5.248080, 4.089333 | 4.803390 | 0.509962 |

| Trial | Phase | Origin | Model | Internal score | Accepted |
|---|---|---|---|---:|---|
| exp_0001 | search | deterministic | linear | 10.755317 | True |
| exp_0002 | search | deterministic | lightgbm | 5.227853 | True |
| exp_0003 | search | qwen | lightgbm | 5.365039 | False |
| exp_0004 | search | deterministic | hist | 5.026616 | True |
| exp_0005 | search | deterministic | extra_trees | 5.621158 | False |
| exp_0006 | search | deterministic | xgboost | 5.312947 | False |
| exp_0007 | search | qwen | hist | 5.197548 | False |
| exp_0008 | search | qwen | hist | 5.168842 | False |
| exp_0009 | search | deterministic | hist | 5.363526 | False |
| exp_0010 | search | deterministic | hist | 5.084222 | False |
| exp_0011 | search | deterministic | hist | 5.026616 | False |
| exp_0012 | search | deterministic | hist | 5.321328 | False |
| exp_0013 | search | deterministic | hist | 5.345511 | False |
| exp_0014 | search | deterministic | hist | 5.064982 | False |
| exp_0015 | search | optuna-tpe | hist | 5.339187 | False |
| exp_0016 | search | optuna-tpe | hist | 7.163388 | False |
| exp_0017 | search | optuna-tpe | lightgbm | 6.346669 | False |
| exp_0018 | search | optuna-tpe | hist | 5.927947 | False |
| exp_0019 | search | optuna-tpe | hist | 5.614583 | False |
| exp_0020 | search | optuna-tpe | lightgbm | 5.163101 | False |
| exp_0021 | search | optuna-tpe | hist | 5.369013 | False |
| exp_0022 | search | optuna-tpe | hist | 5.556534 | False |
| exp_0023 | search | optuna-tpe | lightgbm | 8.370672 | False |
| exp_0024 | search | optuna-tpe | hist | 7.000167 | False |
| exp_0025 | search | optuna-tpe | hist | 6.126959 | False |
| exp_0026 | search | optuna-tpe | lightgbm | 5.072756 | False |
| exp_0027 | confirmation |  | hist | 5.652592 | False |
| exp_0028 | confirmation |  | hist | 5.652592 | False |
| exp_0029 | confirmation |  | lightgbm | 5.248080 | False |
| exp_0030 | confirmation |  | hist | 4.612416 | False |
| exp_0031 | confirmation |  | hist | 4.612416 | False |
| exp_0032 | confirmation |  | lightgbm | 4.089333 | False |
| exp_0033 | refit |  | lightgbm | — | False |

Primary validation acceptance does not guarantee that an external test improves.
This single run is an integration and quality check; it is not a hackathon leaderboard score.
No external metric was supplied to the agent and no post-test model changes were made.
No Kaggle or hackathon submissions were made.

Archive SHA256: `228593d0ddc7a82c21d0201b61c63ab3f163d913bb8644272fb8b9e5e29b14b5`.
Runtime SHA256: `e2f776a1bdeb35a2010a1cba434733fd6edf6d97566cd045a832c1ee27dbb7d9`.
Submission: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-igepsyzz/prepared/inputs/submission.csv`.
Full process log: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-igepsyzz/pipeline.log`.
Run directory: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-igepsyzz`.
