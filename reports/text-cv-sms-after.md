# Full Kaggle pipeline run

Dataset: [team-ai/spam-text-message-classification](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification).
Executed the flat upload as `python3 solution.py task.md`; no agent defaults were overridden.
Local execution uses the existing virtual environment, not a fresh platform Docker image.
The unseen external split was fixed before launch; labels stayed outside the task folder.
Exact duplicate feature groups cannot cross the external train/test boundary.

Status: **completed**. Exit code: 0. Wall time: 120.558 seconds.
Training rows: 4464; external test rows: 1108; original features: 1.
Agent budget: 900.0 seconds; experiments: 53; Qwen requests: 3.
Completed Qwen trials: 1; primary-holdout accepted Qwen trials: 1; full refit: True.

External f1_macro: **0.977125**.

TPE: 30 completed trials; family proposals: {'hist': 10, 'lightgbm': 20}.
Frozen CV: 3 planned folds; 3 fully checked candidates.

| Candidate | Model | Fold scores | Mean | Standard deviation |
|---|---|---|---:|---:|
| exp_0028 | lightgbm | 0.975804, 0.972943, 0.976929 | 0.975226 | 0.001678 |
| exp_0025 | lightgbm | 0.974221, 0.972943, 0.976822 | 0.974662 | 0.001614 |
| exp_0007 | hist | 0.959389, 0.947961, 0.962654 | 0.956668 | 0.006299 |

| Trial | Phase | Origin | Model | Internal score | Accepted |
|---|---|---|---|---:|---|
| exp_0001 | search | deterministic | linear | 0.896206 | True |
| exp_0002 | search | deterministic | lightgbm | 0.945758 | True |
| exp_0003 | search | qwen | lightgbm | 0.966991 | True |
| exp_0004 | search | feature-ablation | lightgbm | 0.964121 | False |
| exp_0005 | search | feature-ablation | lightgbm | 0.948468 | False |
| exp_0006 | search | deterministic | lightgbm | 0.957669 | False |
| exp_0007 | search | deterministic | hist | 0.959389 | False |
| exp_0008 | search | deterministic | extra_trees | 0.946648 | False |
| exp_0009 | search | deterministic | extra_trees | 0.948243 | False |
| exp_0010 | search | deterministic | xgboost | 0.948468 | False |
| exp_0011 | search | deterministic | lightgbm | 0.966991 | False |
| exp_0012 | search | deterministic | lightgbm | 0.967128 | False |
| exp_0013 | search | deterministic | lightgbm | 0.969861 | True |
| exp_0014 | search | deterministic | lightgbm | 0.969735 | False |
| exp_0015 | search | deterministic | lightgbm | 0.965770 | False |
| exp_0016 | search | deterministic | lightgbm | 0.967128 | False |
| exp_0017 | search | optuna-tpe | hist | 0.805813 | False |
| exp_0018 | search | optuna-tpe | lightgbm | 0.940189 | False |
| exp_0019 | search | optuna-tpe | lightgbm | 0.968623 | False |
| exp_0020 | search | optuna-tpe | hist | 0.947110 | False |
| exp_0021 | search | optuna-tpe | lightgbm | 0.964853 | False |
| exp_0022 | search | optuna-tpe | lightgbm | 0.949694 | False |
| exp_0023 | search | optuna-tpe | hist | 0.823287 | False |
| exp_0024 | search | optuna-tpe | lightgbm | 0.971710 | True |
| exp_0025 | search | optuna-tpe | lightgbm | 0.974221 | True |
| exp_0026 | search | optuna-tpe | hist | 0.952782 | False |
| exp_0027 | search | optuna-tpe | lightgbm | 0.973069 | False |
| exp_0028 | search | optuna-tpe | lightgbm | 0.975804 | True |
| exp_0029 | search | optuna-tpe | hist | 0.945118 | False |
| exp_0030 | search | optuna-tpe | lightgbm | 0.971593 | False |
| exp_0031 | search | optuna-tpe | lightgbm | 0.970111 | False |
| exp_0032 | search | optuna-tpe | hist | 0.950264 | False |
| exp_0033 | search | optuna-tpe | lightgbm | 0.970234 | False |
| exp_0034 | search | optuna-tpe | lightgbm | 0.971593 | False |
| exp_0035 | search | optuna-tpe | hist | 0.950185 | False |
| exp_0036 | search | optuna-tpe | lightgbm | 0.969133 | False |
| exp_0037 | search | optuna-tpe | lightgbm | 0.970356 | False |
| exp_0038 | search | optuna-tpe | hist | 0.952035 | False |
| exp_0039 | search | optuna-tpe | lightgbm | 0.971593 | False |
| exp_0040 | search | optuna-tpe | lightgbm | 0.968881 | False |
| exp_0041 | search | optuna-tpe | hist | 0.947338 | False |
| exp_0042 | search | optuna-tpe | lightgbm | 0.971593 | False |
| exp_0043 | search | optuna-tpe | lightgbm | 0.971593 | False |
| exp_0044 | search | optuna-tpe | hist | 0.952035 | False |
| exp_0045 | search | optuna-tpe | lightgbm | 0.945170 | False |
| exp_0046 | search | optuna-tpe | lightgbm | 0.967533 | False |
| exp_0047 | confirmation |  | lightgbm | 0.972943 | False |
| exp_0048 | confirmation |  | lightgbm | 0.972943 | False |
| exp_0049 | confirmation |  | hist | 0.947961 | False |
| exp_0050 | confirmation |  | lightgbm | 0.976929 | False |
| exp_0051 | confirmation |  | lightgbm | 0.976822 | False |
| exp_0052 | confirmation |  | hist | 0.962654 | False |
| exp_0053 | refit |  | lightgbm | — | False |

Primary validation acceptance does not guarantee that an external test improves.
This single run is an integration and quality check; it is not a hackathon leaderboard score.
No external metric was supplied to the agent and no post-test model changes were made.
No Kaggle or hackathon submissions were made.

Archive SHA256: `228593d0ddc7a82c21d0201b61c63ab3f163d913bb8644272fb8b9e5e29b14b5`.
Runtime SHA256: `e2f776a1bdeb35a2010a1cba434733fd6edf6d97566cd045a832c1ee27dbb7d9`.
Submission: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-6ldyl982/prepared/inputs/submission.csv`.
Full process log: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-6ldyl982/pipeline.log`.
Run directory: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-6ldyl982`.
