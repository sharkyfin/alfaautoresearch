# Full Kaggle pipeline run

Dataset: [team-ai/spam-text-message-classification](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification).
Executed the flat upload as `python3 solution.py task.md`; no agent defaults were overridden.
Local execution uses the existing virtual environment, not a fresh platform Docker image.
The unseen external split was fixed before launch; labels stayed outside the task folder.
Exact duplicate feature groups cannot cross the external train/test boundary.

Status: **completed**. Exit code: 0. Wall time: 34.519 seconds.
Training rows: 4464; external test rows: 1108; original features: 1.
Agent budget: 900.0 seconds; experiments: 21; Qwen requests: 3.
Completed Qwen trials: 3; primary-holdout accepted Qwen trials: 1; full refit: True.

External f1_macro: **0.976998**.

| Trial | Phase | Origin | Model | Internal score | Accepted |
|---|---|---|---|---:|---|
| exp_0001 | search | deterministic | linear | 0.936503 | True |
| exp_0002 | search | deterministic | lightgbm | 0.780063 | False |
| exp_0003 | search | qwen | linear | 0.956773 | True |
| exp_0004 | search | feature-ablation | linear | 0.956773 | True |
| exp_0005 | search | feature-ablation | linear | 0.931333 | False |
| exp_0006 | search | feature-ablation | linear | 0.956773 | True |
| exp_0007 | search | deterministic | lightgbm | 0.840260 | False |
| exp_0008 | search | deterministic | hist | 0.767069 | False |
| exp_0009 | search | deterministic | extra_trees | 0.768565 | False |
| exp_0010 | search | deterministic | extra_trees | 0.802626 | False |
| exp_0011 | search | deterministic | xgboost | 0.779792 | False |
| exp_0012 | search | qwen | linear | 0.954644 | False |
| exp_0013 | search | qwen | linear | 0.954644 | False |
| exp_0014 | search | deterministic | linear | 0.977749 | True |
| exp_0015 | search | deterministic | linear | 0.975600 | False |
| exp_0016 | search | deterministic | linear | 0.944497 | False |
| exp_0017 | search | deterministic | linear | 0.900164 | False |
| exp_0018 | confirmation |  | linear | 0.975456 | False |
| exp_0019 | confirmation |  | linear | 0.969226 | False |
| exp_0020 | confirmation |  | linear | 0.945137 | False |
| exp_0021 | refit |  | linear | — | False |

Primary validation acceptance does not guarantee that an external test improves.
This single run is an integration and quality check; it is not a hackathon leaderboard score.
No external metric was supplied to the agent and no post-test model changes were made.
No Kaggle or hackathon submissions were made.

Archive SHA256: `c213bab63badfcf8fc92001c7e3de92c6a7bd333c60931a7e9814a5b1a4ce89c`.
Runtime SHA256: `f2d104a73fac2a01c7c615fee1bcfb0fd18c0faa52a87579b6f1ecd2ca122eb7`.
Submission: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-jqht0q2i/prepared/inputs/submission.csv`.
Full process log: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-jqht0q2i/pipeline.log`.
Run directory: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-jqht0q2i`.
