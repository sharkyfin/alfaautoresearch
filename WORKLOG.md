# ATLAS development state

The entire master task was read and preserved in artifacts/research/MASTER_TASK.md.
PHASE 0 obtained the official PDF, baseline, both requirements variants, task
Markdowns and eight CSVs. CONTRACT.md distinguishes confirmed facts from unknowns.
PHASE 1–5 implemented the deterministic anytime core, six model families,
cost-aware search, typed research recipes, generated feature code, isolated fit
workers, independent evaluation, confirmation, ensemble and full refit.

The final runtime source SHA256 is
`99d4c0b96295fe16e018fba0f83ba27c91b0400854a028d384660208d8bbe244`.
154 tests passed with 13 upstream deprecation warnings in 71.89s; the digest did
not change during the authoritative run (reports/test-state.json). No source
mutations are permitted during release verification without re-running tests.

All paired and public campaign controllers have completed. Their old process
handles in earlier notes are historical and must not be restarted blindly.
The 16-run deterministic/mock paired ablation uses a frozen prototype snapshot;
it does not demonstrate live Qwen value. Final public experiments use snapshot
`0cb87bffe7f686f5f93b14a2067f062a50edcf82d9e70ef57dcdb00ae9faf5ac`:

- Regression: ensemble RMSE 42.728359, confirmation 40.565357, 767.99s.
- Classification: ensemble Macro F1 0.655730, confirmation 0.652610, 810.55s.
- Both two-model full-data refits succeeded. Current output schemas/IDs are
  independently validated by scripts/finalize_release.py.

These are local holdout values; no organizer baseline or leaderboard score is
known. Both public model TaskSpecs remain identical on the final revision after
excluding one new diagnostic field. Later changes and campaign source provenance
are explicitly recorded in reports/RELEASE_AUDIT.md.

Stress evidence: 35 scenarios × 2 seeds, 66 valid outputs, four expected
unsupported cases, no unexpected failures. Among 62 scored nondegenerate cases:
59 gains, one loss, two ties against the fold-fitted dummy. The suite is separate
from short-budget fault-injection, memory, timeout and kill tests.

Both actual public summaries now compact to less than 14 KB and reach the fake
client transport. The full fake-response → validated DSL → generated Python →
actual fit → independent score → feedback loop passes. Those mock checks make zero network calls.

PHASE 6 delivers a source-only **PROVISIONAL** submission.zip. Final authoritative
archive size/SHA/file list and extracted-CLI outcomes live in
reports/release-manifest.json, zip-verification.json and final-release-audit.json.
The deadline simulation uses a fresh directory and successful real experiments:
10800 nominal seconds scaled to 60 actual seconds. It is not a real 3h soak.

External blockers, rechecked 2026-10-20 (Moscow):

1. Organizer OPENAI_BASE_URL, OPENAI_MODEL and OPENAI_API_KEY are unavailable,
   as confirmed by the user. The live loop and live-vs-deterministic equal-budget
   comparison cannot run. Do not use alternative API credentials.
2. Docker is absent from PATH and standard local CLI locations. Exact Linux
   8CPU/32GiB behavior, cgroup/RLIMIT behavior and container network route remain
   unverified. Dockerfile.verify is prepared but unexecuted.

Independent implementation, tests, campaigns and packaging are complete once
final-release-audit.json matches the current runtime and archive hashes. Never
mark the full goal complete while the required external gates remain open.
Next experiment: provision the official Qwen route, check protocol, then repeat
paired equal-budget runs on the frozen suite and verify the actual Linux image.
No platform submission has been made.


## User-authorized OpenRouter development backend

On 2026-10-20 the user requested the best free OpenRouter option and authorized
browser access to obtain a key. Chrome connected to an authenticated account;
OpenRouter onboarding automatically issued a default key. It is stored in the
ignored .env.openrouter.local file (mode 0600) and named ATLAS AutoResearch Free.
No payment details were added or credits purchased. Never print the key value.

Pinned model: minimax/minimax-m3.1-flash:free. The client enforces zero prompt and
completion price ceilings, mandatory low-effort reasoning and 4096 output tokens.
Initial reasoning.enabled=false probe failed with HTTP400; corrected JSON smoke
succeeded. Two subsequent actual research calls cost $0, consumed 12137 tokens,
and generated distinct CatBoost feature/HPO experiments. Both fitted and lost
to the initial champion, so both were rejected. Final ensemble RMSE 0.273152 on
one synthetic holdout, 46.69s elapsed, all full refits passed. This is functional
proof, not an equal-budget quality ablation. Organizer Qwen remains unavailable.

Entry point: scripts/run_openrouter.py. Its explicit research-client injection
never activates automatically in solution.py. scripts/openrouter_client.py and
the key file are excluded from the ZIP. Report: reports/openrouter-live.json.
All 154 tests passed against unchanged runtime digest
99d4c0b96295fe16e018fba0f83ba27c91b0400854a028d384660208d8bbe244.
The official archive was rebuilt and verified with scripts/finalize_release.py;
source and archive digests match the 154-test result.

## Organizer access and corrected platform upload contract

The user supplied organizer access and a new Drive instruction. Chrome read the
complete public Markdown; a verbatim copy is artifacts/official/qwen-access-guide.md.
This explicitly supersedes the prior nested-archive assumption: Boosters copies
only root .py files and root requirements.txt, then runs python3 solution.py task.md
in the task directory. The former archive is withdrawn as incompatible with this
clarification; earlier extraction checks did not simulate that copy filter.

Organizer credentials are stored only in ignored .env.organizer.local (0600).
The first JSON smoke succeeded with qwen/qwen3.8-27b, 122 reported tokens.
A genuine Qwen recipe subsequently compiled and fitted an XGBoost candidate,
which the independent judge accepted. A frozen equal-budget campaign is recorded
under organizer-paired-v1; it is distinct from the completed free OpenRouter
campaign (8 pairs: 3 wins, 3 losses, 2 ties, median normalized gain zero).
No consistent LLM quality advantage is claimed.

The root-only builder stages unchanged own modules as flat Python files plus a
bootstrap that verifies hashes and recreates the package under allowed /tmp.
It bundles the Drive-folder requirements verbatim. The new verifier copies only
eligible root files into a fresh task cwd, checks a fitted checkpoint, sends
SIGTERM and compares the runtime code digest. No keys, datasets or third-party
packages enter the archive. OPENAI_MODEL is now optional, as the official proxy
substitutes the task model. Requests use the explicitly approved httpx 0.28.1.

Offline comparison on four current contexts found a 10KB cap retained objective
and complete executor capabilities while reducing conservative request estimates
from 7.6–8.5K to 6.2–6.6K tokens. The previous 14KB cap sometimes exhausted the
16K per-run token allowance after one response. The smaller cap is undergoing
live validation; old campaign snapshots remain immutable. Full Linux/Python3.12
resolution of the official dependency file succeeded; this is not a full image
installation. The initial wheels-only attempt failed on source-only json2html;
normal metadata resolution succeeded without changing the official requirements.

Lima v2.2.1 was downloaded locally and its release digest verified. A VZ/Ubuntu
24.04 configuration validates with 8CPU/8GiB and no host mounts; the home folder
and key files will not be exposed to the guest. Exact Linux execution remains
pending until a measured linux-verification.json is produced. Physical 32GiB
conditions and the actual platform route cannot be reproduced on this 16GiB Mac.

Organizer-paired-v1 completed: 16 valid runs / 8 pairs, one win, no losses,
seven ties, median normalized improvement zero. Fourteen live requests produced
two fitted Qwen recipes, one accepted; most failures were bounded API timeouts.
Reported costs of received responses sum to 0.9019238703696 (the response schema
does not name currency; timed-out requests may have unreported cost). This is
not a complete billing audit. The updated 10KB/httpx source is now checked in
organizer-paired-v2 on three tasks at equal 240s limits within each pair. V1
used 120s, so differences between campaign versions are not attributable solely
to prompt compaction. No significance or consistent quality gain is claimed.

The live development proxy contradicted the guide's claim that arbitrary model
names are accepted: provided-task-model returned HTTP403 key_model_access_denied.
The optional OPENAI_MODEL fallback now uses the exact documented Qwen model name;
no endpoint or key is embedded. organizer-paired-v3 explicitly passed --omit-model,
confirming the two-variable environment: four API calls, three fitted recipes,
one accepted. Its two equal-budget pairs improved RMSE 0.592150 -> 0.526703 and
Macro F1 0.870364 -> 0.898994. This small sample does not establish significance.

The full suite then found a real signal-handler race: 158 passed / 1 failed,
source unchanged. Immediately after the first CSV appeared, SIGTERM could arrive
before handler installation, yielding -15. run() now installs/restores signal
handlers around the entire orchestration, including initial checkpoint setup.
A deterministic subprocess regression test injects SIGTERM exactly after that
atomic publication; both signal tests pass. The failure log is preserved as
reports/tests-f6cefadcf2.log. A new full suite and current-source live pair precede
packaging and Linux execution; no failed suite is treated as a release pass.

Final corrected-source release gates passed: 160 tests, 13 upstream warnings, 62.97s, unchanged runtime b1fa509ddea5537172c227f44f74a6a8eb1f3b902532f535c52d02a117436e6e.
Final live source pair (v4, two-variable environment): RMSE 0.592150 -> 0.520911, two genuine Qwen fits. Earlier two-task v3 also improved classification, but is kept as its own source snapshot. No pooled significance claim.
Actual flat ZIP b9b4cb7a0c9309a6eab72f893b78d56b4f2ecee85ca0158a23b4b4f70e98df2b passed Linux 6.8 / Python 3.12.15, 8CPU / 6GiB, network disabled: all six families fitted on each of two task types, valid 60-row CSVs, SIGTERM recovery, RLIMIT_AS/RSS/wall-clock guards. VM stopped, zero host mounts. Image build and verification took about 714s. Physical 32GiB and the unreleased platform image/route are not reproduced. Archive size 40247 bytes, 21 root files. No platform submission made.


## 2026-10-24 — ветка monster

Ветка создана от e826f99, включены неизменённый monster.zip (SHA256 30e053296722de373df8b82cef26a976df8220a696aa1a6cd36cb173e2b61189), checksum и последние операционные проверки. Runtime совпадает с проверенной финальной версией; экспериментальный Qwen-кандидат не включён. Последняя проверка: 11 unit-тестов, 16 успешных fit и 0 ошибок workers в четырёх Linux-пробах, отдельный bounded flat-bootstrap/full-refit gate. Эти проверки уже выполнялись до этого коммита; новых ML-бенчмарков при коммите нет. Исходный пользовательский checkout и его незавершённые изменения сохранены. Отправки на платформу нет.
