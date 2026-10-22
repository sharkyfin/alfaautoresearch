# AutoResearch branch restart comparison

Both pipelines received the same original Spaceship Titanic CSVs, three frozen validation splits, eight CPU threads and an explicitly overridden 360-second development budget.
Sources were reconstructed from frozen ZIPs. The API allowance changed from 16000 to 40000 tokens as part of the new policy. This is one stochastic comparison.

| Variant | Wall seconds | Experiments | Branches | Qwen requests | Mean internal accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 77.788 | 36 | 1 | 3 | 0.803866 |
| After | 344.710 | 139 | 5 | 8 | 0.810537 |

Accuracy improved by 0.6672 percentage points, with an improvement on each fold.
The previous ready-made v2 CSV had mean internal accuracy 0.807202 in a separate run with a different configured time budget.
The winning branch transferred features into XGBoost and used 518 trees for full-data refit.

Stop reason: Reserved measured confirmation and refit time.
The search was observed still training at 312.698 seconds, beyond the old 241.2-second cutoff.

All 302 tests passed. Both flat-ZIP official-CLI checkpoint/termination cases passed for the same source hash.
Default ZIP budget remains 10500 seconds, with up to 24 Qwen requests and 120000 tokens. This comparison did not execute a full default-duration run.

Internal validation scores are not official Kaggle scores. No Kaggle or hackathon submission was made.

New CSV: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/spaceship_titanic_submission_v3.csv`.
Current ZIP: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/submission.zip`.
