# Full Kaggle pipeline run

Dataset: [elikplim/concrete-compressive-strength-data-set](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set).
Original data: [I-Cheng Yeh, UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete).

Executed the flat upload as `python3 solution.py task.md`; no agent defaults were overridden.
Local execution uses the existing virtual environment, not a fresh platform Docker image.
The unseen external split was fixed before launch; labels stayed outside the task folder.
Exact duplicate feature groups cannot cross the external train/test boundary.

Status: **completed**. Exit code: 0. Wall time: 64.879 seconds.
Training rows: 820; external test rows: 210; original features: 8.
Agent budget: 900.0 seconds; experiments: 24; Qwen requests: 3.
Completed Qwen trials: 3; primary-holdout accepted Qwen trials: 0; full refit: True.

External rmse: **4.173875**.
MAE: 2.845758; R²: 0.931296; training-mean baseline RMSE: 15.926242.

| Trial | Origin | Model | Internal score | Accepted |
|---|---|---|---:|---|
| exp_0001 | deterministic | linear | 10.446922 | True |
| exp_0002 | deterministic | lightgbm | 4.603486 | True |
| exp_0003 | qwen | lightgbm | 4.917713 | False |
| exp_0004 | deterministic | hist | 4.718862 | False |
| exp_0005 | deterministic | extra_trees | 4.695538 | False |
| exp_0006 | deterministic | xgboost | 4.395054 | True |
| exp_0007 | qwen | xgboost | 4.422757 | False |
| exp_0008 | qwen | xgboost | 4.435095 | False |
| exp_0009 | deterministic | xgboost | 4.283763 | True |
| exp_0010 | deterministic | xgboost | 4.353880 | False |
| exp_0011 | deterministic | xgboost | 4.283763 | False |
| exp_0012 | deterministic | xgboost | 4.417929 | False |
| exp_0013 | deterministic | xgboost | 4.810452 | False |
| exp_0014 | deterministic | xgboost | 4.791424 | False |
| exp_0015 | deterministic | xgboost | 4.703158 | False |
| exp_0016 | deterministic | xgboost | 4.509039 | False |
| exp_0017 | deterministic | xgboost | 4.748137 | False |
| exp_0018 | deterministic | xgboost | 4.574697 | False |
| exp_0019 | deterministic | xgboost | 4.616977 | False |
| exp_0020 |  | xgboost | 5.317118 | False |
| exp_0021 |  | xgboost | 5.317118 | False |
| exp_0022 |  | xgboost | 5.358366 | False |
| exp_0023 |  | xgboost | — | False |
| exp_0024 |  | xgboost | — | False |

Primary validation acceptance does not guarantee that an external test improves.
This single run is an integration and quality check; it is not a hackathon leaderboard score.
No external metric was supplied to the agent and no post-test model changes were made.
No Kaggle or hackathon submissions were made.

Archive SHA256: `c213bab63badfcf8fc92001c7e3de92c6a7bd333c60931a7e9814a5b1a4ce89c`.
Runtime SHA256: `f2d104a73fac2a01c7c615fee1bcfb0fd18c0faa52a87579b6f1ecd2ca122eb7`.
Submission: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-iuj74mxd/prepared/inputs/submission.csv`.
Full process log: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-iuj74mxd/pipeline.log`.
Run directory: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/runs/kaggle-full-iuj74mxd`.
