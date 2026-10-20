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
