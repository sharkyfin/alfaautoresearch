# Qwen feature formulas

Qwen can propose task-specific named features in `feature_expressions`. The same
formulas run on training, validation and test data. Formula execution is supported
by all five active model families: linear, HistGradientBoosting, ExtraTrees,
LightGBM and XGBoost. CatBoost is excluded from search and research capabilities.

Example research proposal, assuming the indicated input columns exist:

```json
{
  "model_family": "lightgbm",
  "params": {"iterations": 500, "num_leaves": 31},
  "feature_mode": "append",
  "feature_expressions": [
    {
      "name": "bytes_per_packet",
      "expression": "bytes / (packets + 1)",
      "reason": "Packet size may distinguish traffic patterns"
    },
    {
      "name": "large_packet",
      "expression": "where(bytes_per_packet > 1000, 1, 0)"
    },
    {
      "name": "relative_packet_size",
      "expression": "bytes_per_packet / maximum(group_mean(bytes_per_packet, service), 1)"
    }
  ],
  "statement": "Test packet size and its deviation from service-level norms",
  "max_runtime_sec": 120
}
```

Only input feature columns and earlier named formulas can be referenced. Use
`col("column with spaces")` when a column is not a Python identifier. Target and
ID references, forward references, cycles and duplicate output names are rejected.

Research proposals append features to the accepted parent by default. A formula
with an existing name updates that formula in place. `feature_mode: "replace"`
performs an explicit ablation using only the supplied feature lists. Deterministic
local search also retains accepted named formulas and earlier legacy operations.

Feature hypotheses now hold the champion's model family, parameters, target
transform, split and seed fixed. Requested model changes are recorded and ignored
for these trials; propose model tuning separately. Change at most four
formulas/legacy operations per hypothesis. A successful group may receive up to
three dependency-aware ablation fits within the same run budget. The total limit
of 16 formulas is unchanged. See [AUTORESEARCH.md](AUTORESEARCH.md).

## Supported expressions

- Arithmetic: `+`, `-`, `*`, `/`, `%`, `**` with a constant exponent in [-4, 4].
- Comparisons, `and`, `or`, `not`, `&`, `|`, and `a if condition else b`.
- Numeric functions: `abs`, `log1p`, `signed_log1p`, `sqrt`, `clip(x, low, high)`,
  `minimum`, `maximum`, `fillna`, `isna`, `where(condition, a, b)`.
- Dates: `year`, `month`, `day`, `dayofweek`, `hour`,
  `days_between(later, earlier)`.
- Text: `length`, `word_count`, `digit_count`, `lower`, `concat(a, b[, c])`.
- Learned statistics: `frequency(group)`, `group_count(group)`,
  `group_mean(value, group)`, `group_median(value, group)`,
  `group_std(value, group)`.

Division/modulo by near-zero denominators produces missing values, handled by the
normal model preprocessing. `log1p` requires values greater than -1; use
`signed_log1p` for signed values. Invalid numeric results become missing. Numeric
outputs are clipped to finite magnitudes. Dates are interpreted in UTC, elapsed
time is measured in days, and generated strings are limited to 4096 characters.

Each statistics mapping is fitted only on the experiment's training fold and
reused unchanged on validation/test. Unknown groups receive zero for
frequency/count/std and the training-fold global median for mean/median.
At final refit, mappings are relearned from the full training data.
Target encoding is not implemented by these functions.

## Execution and limits

Formulas are parsed and interpreted through a whitelist of AST nodes. They are
never evaluated with `eval`, `exec` or `pandas.eval`; there is no attribute access,
indexing, import, file operation or arbitrary Python statement in this format.
Existing worker time/memory limits continue to apply.

A recipe supports up to 16 formulas, 768 characters per expression, 96 AST nodes
per expression, 512 nodes overall, 12 nesting levels and 4096 UTF-8 bytes across
expressions/reasons. Invalid recipes are recorded as rejected hypotheses. A
formula is not evidence of improvement until the fixed validation metric improves.

## Local checks

From the repository root with its Python 3.12 environment:

The official entrypoint and generated upload use a fast 900-second (15-minute)
budget per task by default. It may finish earlier. Platform queue time is separate.
Development wrappers accept `--budget 10500` for a longer local search; their own
short defaults are unchanged.

```sh
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/feature_smoke.py
.venv/bin/python scripts/feature_smoke.py --live
```

The offline smoke fits a real baseline and feature-enhanced model. The live smoke
makes at most two organizer Qwen requests, validates the returned formulas, trains
a worker and writes a checked `submission.csv`. It reads the ignored
`.env.organizer.local` file, never forwards the API key to workers, and saves
diagnostics under a new `runs/feature-smoke-*` directory.

The synthetic target deliberately contains a multiplicative interaction and the
live diagnostic explicitly suggests testing that interaction. This verifies the
API-to-feature-to-model path, not autonomous discovery or competition quality.
Use the real public tasks for quality comparisons under equal time budgets.

Build and validate the upload separately:

```sh
.venv/bin/python scripts/package_submission.py
.venv/bin/python scripts/verify_zip.py
```

The official requirements file in the upload stays byte-for-byte unchanged.
It still lists CatBoost because it is provided by the organizers; the solution
does not import, train or propose that backend.
