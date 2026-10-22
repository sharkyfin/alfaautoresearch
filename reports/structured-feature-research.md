# Structured feature research on Spaceship Titanic

Same original training/test CSV hashes and the same three frozen internal validation splits. Agent defaults were unchanged.

| Variant | Mean internal accuracy | Fold 1 | Fold 2 | Fold 3 | Seconds | Completed Qwen fits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 0.796618 | 0.800828 | 0.788130 | 0.800897 | 109.561 | 0 |
| Syntax and string operations | 0.798574 | 0.805661 | 0.789855 | 0.800207 | 76.009 | 3 |
| With training-fitted separator repair | 0.807202 | 0.814291 | 0.796756 | 0.810559 | 89.938 | 2 |

Final improvement: 1.0584 percentage points, positive on all three folds.
Final model: LightGBM, 159 trees, full-data refit. Selected generated features: deck, side and total spending. Cabin number was removed by controlled ablation.
Two of three Qwen proposals reached training. The third was rejected for invalid formula syntax; fallback search continued.
291 tests passed; the flat ZIP passed both checkpoint/termination scenarios. See structured-feature-research-checks.json for hashes and validation.

Official Kaggle score and leaderboard rank are pending manual submission. This is a single local comparison, not a guarantee for unseen hackathon tasks.

New manual-upload CSV: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/spaceship_titanic_submission_v2.csv`.
Current agent ZIP: `/Users/nosovdanil/PycharmProjects/1/alfaautoresearch/submission.zip`.
