"""Match actual live/fixed trials to same-source A0 observations across campaigns."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.report_performance_v2 import effect, stats

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'reports/performance_v2'


def main():
    names = ['p3h-broad-120', 'api-admission-live']
    reports = []
    for name in names:
        path = DIRECTORY / (name + '.json')
        if not path.exists():
            print('Matched Qwen panel not complete; no result written.')
            return
        value = json.loads(path.read_text())
        if value.get('status') != 'completed':
            print('Matched Qwen panel not complete; no result written.')
            return
        reports.append(value)
    reference = {(row['case'], row['seed']): row for row in reports[0]['runs']
                 if row['arm'] == 'candidate' and row['mode'] == 'deterministic'}
    comparisons = []
    records = []
    for row in reports[1]['runs']:
        before = reference[(row['case'], row['seed'])]
        assert row['mode'] in {'live-agent', 'fixed-allocation'}
        for key in ['source_sha', 'zip_sha', 'budget_seconds', 'split_sha256', 'input_sha256']:
            if before[key] != row[key]:
                raise ValueError('Qwen panel mismatch: ' + key)
        comparisons.append(effect(before, row, row['mode'] + ' vs matched A0'))
        records.append({key: row.get(key) for key in
                        ['case', 'seed', 'mode', 'qwen_calls', 'qwen_fits', 'qwen_accepted',
                         'qwen_tokens', 'wall_seconds', 'external_metric', 'full_refit', 'qwen_events']})
    summaries = {mode: stats([row for row in comparisons if row['comparison'] == mode + ' vs matched A0'])
                 for mode in ['live-agent', 'fixed-allocation']}
    live = [row for row in records if row['mode'] == 'live-agent']
    counts = {key: sum(row.get(key) or 0 for row in live)
              for key in ['qwen_calls', 'qwen_fits', 'qwen_accepted', 'qwen_tokens']}
    result = {'schema_version': 2, 'derived_from_executed_campaigns': names,
              'note': 'A0 reused from broad observations, not a second run. No derived rows added to benchmark_matrix.',
              'source_sha': live and reports[1]['versions']['candidate']['source_sha'],
              'statistics': summaries, 'live_counts': counts, 'pairs': comparisons, 'api_observations': records}
    (DIRECTORY / 'matched_qwen_effect.json').write_text(json.dumps(result, indent=2) + '\n')
    fields = sorted({key for row in comparisons for key in row})
    with (DIRECTORY / 'matched_qwen_effect.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    lines = ['# Same-source Qwen admission follow-up', '',
             'A0 is reused from the completed P3h broad comparison. A1 live-Qwen and A2 fixed allocation '
             'were executed separately on the same three datasets and two seeds. Source, archive, input, split '
             'and 120-second cap identities are asserted equal before matching. No extra A0 execution is claimed.', '',
             '| Dataset | Seed | Comparison | A0 | Candidate | Relative effect | A0 / candidate seconds |',
             '|---|---:|---|---:|---:|---:|---:|']
    for row in comparisons:
        def fmt(value):
            return '—' if value is None else f'{value:.9g}'
        lines.append(f"| {row['case']} | {row['seed']} | {row['comparison']} | "
                     f"{fmt(row.get('baseline_score'))} | {fmt(row.get('candidate_score'))} | "
                     f"{fmt(100*row['relative_gain'])+'%' if 'relative_gain' in row else row['outcome']} | "
                     f"{fmt(row.get('baseline_seconds'))} / {fmt(row.get('candidate_seconds'))} |")
    lines += ['', '## Observed effects', '', '```json', json.dumps(summaries, indent=2), '```', '',
              'Actual live usage: `' + json.dumps(counts) + '`. Internal acceptance is not external improvement.', '',
              'P3g/P3h admission reserves a measured API estimate, a real probe and controls/refit before calling. '
              'Skipped-finalization events are not counted as HTTP requests. This short-budget, three-dataset '
              'panel does not establish a 10500-second or universal LLM benefit. API sampling, latency and '
              'wall-clock trial admission can change the search trajectory; keep all negative cases.', '',
              'Earlier P3e study is retained in `llm_value_ablation.md`; its different source is not used as '
              'the deterministic denominator here. Full event histories: `research_trace.json`.', '']
    (DIRECTORY / 'matched_qwen_effect.md').write_text('\n'.join(lines))
    print(json.dumps({'matched_pairs': len(comparisons), 'live_counts': counts}))


if __name__ == '__main__':
    main()
