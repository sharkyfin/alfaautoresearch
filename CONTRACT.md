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

## Latest organizer clarification (takes precedence)

Source: [official instruction](https://drive.google.com/file/d/17yxOkrdFHeQFX29iCz3f6noMs3l9cW2v/view),
read in Chrome and downloaded verbatim as artifacts/official/qwen-access-guide.md.

- All uploaded files must be at ZIP root: solution.py, own .py modules, official requirements.txt.
- Only root .py and requirements.txt are copied into the task directory. Nested atlas/ is discarded.
- Exact invocation: python3 solution.py task.md, cwd is the task directory.
- Runtime may create directories in the task folder and /tmp. Our flat bootstrap reconstructs only our own unchanged sources there; no packages or binaries are bundled.
- ZIP limit 100MB; 8CPU, 32GB RAM, 10800 seconds.
- Environment supplies OPENAI_BASE_URL and OPENAI_API_KEY. OPENAI_MODEL is optional. The guide promises model rewriting; a real development call with a generic name returned HTTP403 key_model_access_denied. Runtime therefore defaults to the guide's qwen/qwen3.8-27b name, still allowing an environment override.
- Local development endpoint https://changellenge.up.railway.app/v1, model qwen/qwen3.8-27b. These values are NOT hardcoded in runtime.
- Shared key budget 600RUB/day, reset 03:00 Moscow. Platform submission quota is separate.
- User supplied organizer credentials; local JSON smoke succeeded and a real Qwen proposal compiled, fitted and passed independent scoring. See reports/organizer-protocol-smoke.json and organizer-paired reports.

The former nested ZIP was incompatible with this clarification and is superseded.
New verification must simulate the actual root-file copy filter, not extract and run nested modules.

## Unverified environment details

Actual organizer image and platform network route, disk quota, cgroup layout,
private probability schema and additional target types. A local Linux VM test
cannot prove the platform route or physical 32GiB conditions on a 16GiB host.
Current local execution gates are recorded in reports/final-release-audit.json.
Public downloads used the normal Drive download-confirmation form.

## Public Markdown contract (CONFIRMED FROM FILE)

Downloaded both official task.md documents and all eight CSVs using public Drive links.
Both require submission.csv in the Markdown directory, UTF-8/comma/no index.
Regression inputs are under regression/participants; classification inputs under multiclass/participants.
Both outputs have id then target; finite values / valid original labels, test row order, unique IDs.
Public regression explicitly describes contemporaneous observed weather: not a forecasting task.
Public classification primary Macro F1 is distinguished from auxiliary accuracy and weighted F1.

## Development backends

The earlier organizer-key blocker is resolved by the latest user message.
Local key files remain ignored and are excluded from runtime workers and ZIP.

On 2026-10-20 the user separately authorized a free OpenRouter backend for
development and browser access to their account to obtain a key. OpenRouter's
initial workspace setup automatically issued a default key, saved locally in
ignored `.env.openrouter.local` with mode 0600. No payment details were added.
The pinned development model is minimax/minimax-m3.1-flash:free; request price
limits are zero and there is no paid-model fallback. scripts/run_openrouter.py
injects this client explicitly; solution.py does not import it or read its key.
OpenRouter is not evidence that the organizer Qwen route works or is approved.

## Requirements variants

The Drive folder contains a second requirements.txt (ID 1M4-df5jF4Zfni5R9_cmnTWQ1913ZlQLb).
It adds langchain-openai and omits requests/aiohttp compared with the PDF-linked
file. All ATLAS runtime dependencies are in the intersection, at identical pins.
Both versions are preserved under artifacts/official; neither is silently replaced.

The ZIP uses the folder version verbatim as root requirements.txt. A Linux/Python3.12
resolution of that complete file succeeded via uv pip compile (reports/official-requirements-resolution.log);
this is dependency-resolution evidence, not an installation of the full organizer image.
