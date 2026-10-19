# Primary-source checks during implementation

Read 2026-10-19 in development. Runtime does not fetch these resources.

- [LADS](https://github.com/sb-ai-lab/LADS): hybrid code-generation/AutoML idea; no framework code copied into ATLAS.
- [AIDE](https://github.com/wecoai/aideml): history/branch selection as design input, reduced to bounded champion/challenger search here.
- [autoresearch](https://github.com/karpathy/autoresearch): fixed experiment objective and keep/reject discipline.
- [FLAML task-oriented AutoML](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/): explicit time budgets and cost-aware search; not an extra runtime dependency.
- [AutoGluon essentials](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html): portfolio/ensemble reference; foundation presets are not used.
- [sklearn 1.5 pitfalls](https://scikit-learn.org/1.5/common_pitfalls.html): preprocessing fitted to training folds.
- [CatBoost common parameters](https://catboost.ai/docs/en/references/training-parameters/common): version/API checks; ATLAS uses explicitly computed sample weights, never class_weights='balanced'.
- [MLE-bench](https://github.com/openai/mle-bench): repeated-run evaluation reference; our synthetic and public runs do not imply a benchmark leaderboard result.
- [OpenFE](https://github.com/IIIS-Li-Group/OpenFE): bounded feature exploration design input; no package imported.

These are design references, not evidence that ATLAS achieves their reported scores. Organizer PDF/baseline/requirements and actual local measurements govern compatibility and claims.
