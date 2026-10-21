# Supported task surface and private-task risks

| Surface | Supported behavior | Boundary |
|---|---|---|
| Input contract | UTF-8 comma CSV + UTF-8 task Markdown; paths relative to Markdown | Referenced paths must remain within the task directory |
| Targets | One supervised regression/binary/multiclass target | Multiple/contradictory targets are explicitly unsupported |
| Labels | Original numeric/nonconsecutive/string/Unicode labels; fixed class index mapping | Classes are learned only from train; novel test classes cannot be learned |
| IDs | Dictionary ID roles take priority; unique shared sample/test columns are a fallback; leading zeros preserved; one-to-one join | Ambiguous non-unique composite IDs require dictionary roles |
| Output | submission.csv beside task.md; sample column order, finite values, permitted labels, row count and sample ID order | Historical output examples in prose do not change the platform filename |
| Regression metrics | RMSE, MSE, MAE, R2, RMSLE | RMSLE requires nonnegative targets; very tiny folds can make R2 undefined |
| Hard-label metrics | Accuracy, balanced accuracy, binary/macro/micro/weighted F1 | Macro F1 fixes the train class set and zero_division=0 |
| Probabilistic metrics | Binary/multiclass ROC AUC, logloss; binary AP | Multiclass AUC defaults to OVR/macro unless described otherwise; positive class defaults to index 1 unless specified |
| Probability CSV | Binary single column, or one exact class-name column per class | Other naming conventions / multilabel layouts are unsupported |
| Missing/unseen data | Fold-fitted imputation/encoding; inf sanitization; unknown categories; numeric strings | No arbitrary locale delimiter or Excel format loader |
| Dates/text/IP | Datetime components and cycles, TF-IDF for up to two text fields, bounded IP parsing DSL | Numeric timestamp units and complex text semantics are not universally inferred |
| Validation | IID, rare-class stratification, explicit forecasting, disjoint groups with evidence, duplicate grouping | Mixed seen/new entity test populations and undocumented temporal intent remain inference risks |
| Research | Five model families, named nested formulas, training-fold statistics, early Qwen proposal, independent metric gate and retained research feedback | No unrestricted Python generation, pretrained models or external datasets |
| Ensembles | Common holdout probabilities/predictions; fixed small convex weight grid; second-split confirmation | No in-sample stacking; no claim of exhaustive nested-CV optimization |
| Failures | Model errors/timeouts/RSS limit, Qwen faults, worker kill and SIGTERM preserve earlier CSV | Unavailable/corrupt inputs or unwritable filesystem cannot be guaranteed |
| Unknown metric | Explicit diagnostic plus RMSE/Macro F1 fallback | No claim that the unknown formula is optimized |
| LLM | Official env contract; real organizer Qwen formula smoke recorded; complete flat-upload CLI checked through a local protocol stub | Exact platform network route is external; OpenRouter development client is excluded from the competition ZIP |
| Environment | Python 3.12 on macOS; approved library pins; 8-thread/24GiB-worker/900-second default controls | Current organizer Linux Docker image is unavailable locally; historical Linux checks belong to their recorded archive SHA |

Synthetic coverage includes 35 scenarios × 2 seeds. This demonstrates tested
behaviors and controlled degradation, not exhaustive coverage of hidden tasks.
No leaderboard score, private labels, statistical significance or guaranteed win
is inferred from local results.
