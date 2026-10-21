# Full Kaggle pipeline run

Dataset: [camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices).
Executed the flat upload as `python3 solution.py task.md`; no agent defaults were overridden.
Local execution uses the existing virtual environment, not a fresh platform Docker image.
The unseen external split was fixed before launch; labels stayed outside the task folder.
Exact duplicate feature groups cannot cross the external train/test boundary.

Status: **completed**. Exit code: 0. Wall time: 334.35 seconds.
Training rows: 16512; external test rows: 4128; original features: 9.
Agent budget: 10500.0 seconds; experiments: 68; Qwen requests: 2.
Completed Qwen trials: 1; primary-holdout accepted Qwen trials: 1; full refit: True.

External rmse: **42476.658069**.
MAE: 27647.624871; R²: 0.862313; training-mean baseline RMSE: 114485.635431.

TPE: 45 completed trials; family proposals: {'xgboost': 15, 'lightgbm': 30}.
Frozen CV: 3 planned folds; 3 fully checked candidates.

| Candidate | Model | Fold scores | Mean | Standard deviation |
|---|---|---|---:|---:|
| exp_0042 | lightgbm | 43807.371185, 44317.564169, 44163.278536 | 44096.071297 | 213.638058 |
| exp_0048 | lightgbm | 43906.356604, 44386.239127, 44524.375091 | 44272.323607 | 264.851244 |
| exp_0055 | xgboost | 45534.710611, 46142.744047, 45875.774941 | 45851.076533 | 248.842217 |

| Trial | Phase | Origin | Model | Internal score | Accepted |
|---|---|---|---|---:|---|
| exp_0001 | search | deterministic | linear | 69614.337233 | True |
| exp_0002 | search | deterministic | lightgbm | 46465.015429 | True |
| exp_0003 | search | qwen | lightgbm | 45657.545489 | True |
| exp_0004 | search | feature-ablation | lightgbm | 46597.095967 | False |
| exp_0005 | search | feature-ablation | lightgbm | 45522.043841 | True |
| exp_0006 | search | feature-ablation | lightgbm | 45424.974299 | True |
| exp_0007 | search | deterministic | hist | 46579.575873 | False |
| exp_0008 | search | deterministic | extra_trees | 52098.592544 | False |
| exp_0009 | search | deterministic | xgboost | 46469.453389 | False |
| exp_0010 | search | deterministic | lightgbm | 45633.777655 | False |
| exp_0011 | search | deterministic | lightgbm | 45056.776497 | True |
| exp_0012 | search | deterministic | lightgbm | 44555.008567 | True |
| exp_0013 | search | deterministic | lightgbm | 44724.918944 | False |
| exp_0014 | search | deterministic | lightgbm | 44912.429136 | False |
| exp_0015 | search | deterministic | lightgbm | 44605.795002 | False |
| exp_0016 | search | optuna-tpe | xgboost | 46005.510918 | False |
| exp_0017 | search | optuna-tpe | lightgbm | 45356.873222 | False |
| exp_0018 | search | optuna-tpe | lightgbm | 44592.409928 | False |
| exp_0019 | search | optuna-tpe | xgboost | 46013.867441 | False |
| exp_0020 | search | optuna-tpe | lightgbm | 45411.066854 | False |
| exp_0021 | search | optuna-tpe | lightgbm | 44274.939878 | True |
| exp_0022 | search | optuna-tpe | xgboost | 47873.400835 | False |
| exp_0023 | search | optuna-tpe | lightgbm | 44861.625197 | False |
| exp_0024 | search | optuna-tpe | lightgbm | 44171.728654 | True |
| exp_0025 | search | optuna-tpe | xgboost | 46127.969445 | False |
| exp_0026 | search | optuna-tpe | lightgbm | 44258.380991 | False |
| exp_0027 | search | optuna-tpe | lightgbm | 45274.180875 | False |
| exp_0028 | search | optuna-tpe | xgboost | 46339.671338 | False |
| exp_0029 | search | optuna-tpe | lightgbm | 44793.146927 | False |
| exp_0030 | search | optuna-tpe | lightgbm | 44365.365971 | False |
| exp_0031 | search | optuna-tpe | xgboost | 46032.210247 | False |
| exp_0032 | search | optuna-tpe | lightgbm | 44566.516582 | False |
| exp_0033 | search | optuna-tpe | lightgbm | 44461.401503 | False |
| exp_0034 | search | optuna-tpe | xgboost | 49510.711981 | False |
| exp_0035 | search | optuna-tpe | lightgbm | 44597.226732 | False |
| exp_0036 | search | optuna-tpe | lightgbm | 44268.168281 | False |
| exp_0037 | search | optuna-tpe | xgboost | 45558.612798 | False |
| exp_0038 | search | optuna-tpe | lightgbm | 45034.180692 | False |
| exp_0039 | search | optuna-tpe | lightgbm | 44109.710769 | True |
| exp_0040 | search | optuna-tpe | xgboost | 46197.671711 | False |
| exp_0041 | search | optuna-tpe | lightgbm | 44353.949803 | False |
| exp_0042 | search | optuna-tpe | lightgbm | 43807.371185 | True |
| exp_0043 | search | optuna-tpe | xgboost | 46334.163857 | False |
| exp_0044 | search | optuna-tpe | lightgbm | 44271.346060 | False |
| exp_0045 | search | optuna-tpe | lightgbm | 44300.004086 | False |
| exp_0046 | search | optuna-tpe | xgboost | 45599.205211 | False |
| exp_0047 | search | optuna-tpe | lightgbm | 43936.597864 | False |
| exp_0048 | search | optuna-tpe | lightgbm | 43906.356604 | False |
| exp_0049 | search | optuna-tpe | xgboost | 45567.708683 | False |
| exp_0050 | search | optuna-tpe | lightgbm | 44660.619173 | False |
| exp_0051 | search | optuna-tpe | lightgbm | 44357.435327 | False |
| exp_0052 | search | optuna-tpe | xgboost | 45574.237205 | False |
| exp_0053 | search | optuna-tpe | lightgbm | 44611.148837 | False |
| exp_0054 | search | optuna-tpe | lightgbm | 44445.658591 | False |
| exp_0055 | search | optuna-tpe | xgboost | 45534.710611 | False |
| exp_0056 | search | optuna-tpe | lightgbm | 44415.978898 | False |
| exp_0057 | search | optuna-tpe | lightgbm | 45293.068306 | False |
| exp_0058 | search | optuna-tpe | xgboost | 45644.867037 | False |
| exp_0059 | search | optuna-tpe | lightgbm | 44153.037195 | False |
| exp_0060 | search | optuna-tpe | lightgbm | 44114.342336 | False |
| exp_0061 | confirmation |  | lightgbm | 44317.564169 | False |
| exp_0062 | confirmation |  | lightgbm | 44386.239127 | False |
| exp_0063 | confirmation |  | xgboost | 46142.744047 | False |
| exp_0064 | confirmation |  | lightgbm | 44163.278536 | False |
| exp_0065 | confirmation |  | lightgbm | 44524.375091 | False |
| exp_0066 | confirmation |  | xgboost | 45875.774941 | False |
| exp_0067 | refit |  | lightgbm | — | False |
| exp_0068 | refit |  | xgboost | — | False |

Primary validation acceptance does not guarantee that an external test improves.
This single run is an integration and quality check; it is not a hackathon leaderboard score.
No external metric was supplied to the agent and no post-test model changes were made.
No Kaggle or hackathon submissions were made.

Archive SHA256: `95080f8dd3a651c6762df6933a82091f89abc708cd6d1797363b49a102a64bc9`.
Runtime SHA256: `110869ae56302830a4d920ad4bde9d06e9c7b2ac45a3e97aa6ab96252331c507`.
Submission: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-66pom43p/prepared/inputs/submission.csv`.
Full process log: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-66pom43p/pipeline.log`.
Run directory: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-66pom43p`.
