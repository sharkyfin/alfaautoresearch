# Runtime contract reconnaissance

## CONFIRMED FROM FILE

Sources: artifacts/official/baseline.py, requirements.txt, retrieved 2026-10-19.

- Baseline CLI: `python baseline.py [task.md]`, default task.md; resolves Markdown path.
- Markdown is read as UTF-8 and passed verbatim to the agent.
- Code tool runs Python subprocesses with cwd equal to task Markdown directory.
- Tool timeout 1800 seconds; new process per invocation; files persist.
- Parent LLM uses AsyncOpenAI(), OPENAI_MODEL; OPENAI_API_KEY and OPENAI_BASE_URL consumed by OpenAI SDK.
- Worker env is restricted to PATH and PYTHONIOENCODING. API keys are not forwarded.
- Baseline has no fixed CSV filename or return object: output instructions are supplied by task Markdown.
- MCP imports are used in baseline but `mcp` is absent from supplied requirements. ATLAS avoids MCP runtime dependency.
- Approved CPU backends: sklearn 1.5.2, CatBoost 1.2.8, LightGBM 4.6.0, XGBoost 2.1.4; numpy 1.26.4, pandas 2.2.3, scipy 1.12.0.
- Official tool smoke: `python3 scripts/baseline_smoke.py`; verified cwd/file persistence/early return and environment filtering without network or LLM.

## CONFIRMED FROM PDF

Local official rules.pdf, 20 pages, extracted rules.txt.
Python 3.12, Docker, 8 CPU, 32 GB RAM, 10800s/task. ZIP root solution.py.
No external training data, pretrained weights or alternative keyed APIs; no ordinary internet.
Three submissions/day, reset midnight Moscow. No submission made by this project.
The prediction file present at timeout is evaluated; early completion permitted.
Task inputs: train.csv, test.csv, data_dictionary.csv, sample_submission.csv.
Public metrics RMSE and macro F1; private metrics unknown.

## UNVERIFIED

- Evaluation endpoint/model values and whether Qwen route is reachable in evaluation container.
- Maximum disk quota, cgroup layout, extra platform wrapper arguments and whether script.py on PDF p17 is a typo.
- Exact private probability-column conventions and additional target types.

## BLOCKED

No API credentials or actual endpoint/model values provided. Live Qwen verification and true live-vs-deterministic ablation cannot run yet. Deterministic and mock-agent modes will be tested independently.

## Release policy

Until all release gates pass, submission.zip is explicitly provisional. Public file download succeeded using the normal Drive download-confirmation form; access restrictions were not bypassed.

## Public Markdown contract (CONFIRMED FROM FILE)

Downloaded both official task.md documents and all eight CSVs using public Drive links.
Both require submission.csv in the Markdown directory, UTF-8/comma/no index.
Regression inputs are under regression/participants; classification inputs under multiclass/participants.
Both outputs have id then target; finite values / valid original labels, test row order, unique IDs.
Public regression explicitly describes contemporaneous observed weather: not a forecasting task.
Public classification primary Macro F1 is distinguished from auxiliary accuracy and weighted F1.

## API blocker confirmed by user

The user confirmed that organizer Qwen settings are not yet available. No live
API calls have been attempted; no alternative keys/services will be used.

## Requirements variants

The Drive folder contains a second requirements.txt (ID 1M4-df5jF4Zfni5R9_cmnTWQ1913ZlQLb).
It adds langchain-openai and omits requests/aiohttp compared with the PDF-linked
file. All ATLAS runtime dependencies are in the intersection, at identical pins.
Both versions are preserved under artifacts/official; neither is silently replaced.
