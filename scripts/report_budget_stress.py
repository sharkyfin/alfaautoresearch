"""Report observed budget curves; never relabel a short run as a three-hour soak."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'reports/performance_v2'


def main():
    rows = []
    for path in sorted(DIRECTORY.glob('budget-*.json')):
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get('status') != 'completed':
            continue
        for row in value.get('runs', []):
            if not isinstance(row, dict) or 'case' not in row:
                continue
            rows.append({**row, 'campaign': path.stem,
                         'configured_budget_fraction_used': row.get('wall_seconds', 0) / row['budget_seconds']})
    if not rows:
        print('No completed budget campaigns; no stress result produced.')
        return
    rows.sort(key=lambda row: (row['case'], row['seed'], row['budget_seconds'], row['arm']))
    fields = ['campaign', 'case', 'seed', 'arm', 'budget_seconds', 'wall_seconds',
              'configured_budget_fraction_used', 'external_metric', 'metric', 'status',
              'first_trained_checkpoint_seconds', 'cpu_seconds', 'peak_memory_mb',
              'fit_success', 'experiments', 'control_folds_completed', 'full_refit',
              'champion_family', 'source_sha', 'zip_sha', 'split_sha256']
    with (DIRECTORY / 'budget_stress.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Measured budget curves and stress', '',
             'All points below come from completed runs. Configured caps and actual wall times are different quantities. '
             'Early convergence is not a continuous three-hour soak. Ordinary platform CLI default is 10500 seconds, '
             'with 300 seconds before the external 10800-second limit. Development guards include imports, fits, '
             'confirmation, refit, output and API delays if configured.', '',
             '| Dataset | Seed | Arm | Cap s | Wall s | Used | Metric | Outer | First trained s | Peak MiB | Refit | Status |',
             '|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|']
    def fmt(value):
        return '—' if value is None else f'{value:.9g}'
    for row in rows:
        lines.append(f"| {row['case']} | {row['seed']} | {row['arm']} | {row['budget_seconds']:g} | "
                     f"{row.get('wall_seconds', 0):.2f} | {100*row['configured_budget_fraction_used']:.1f}% | "
                     f"{row['metric']} | {fmt(row.get('external_metric'))} | "
                     f"{fmt(row.get('first_trained_checkpoint_seconds'))} | {fmt(row.get('peak_memory_mb'))} | "
                     f"{row.get('full_refit')} | {row['status']} |")
    lines += ['', '## Search termination', '']
    for row in rows:
        search = row.get('adaptive_search') or {}
        lines.append(f"- {row['case']}, {row['arm']}, cap {row['budget_seconds']:g}: "
                     f"{search.get('research_steps')} research steps, {search.get('completed_trials')} completed TPE trials; "
                     f"stop reason: {search.get('stop_reason')!r}; terminal state: {row.get('terminal_status')!r}.")
    lines += ['', '## Limits', '',
              'The organizer training-data holdout has no access to official test truth. The resource panel uses '
              'the local host, eight configured compute threads and an 8192-MiB aggregate guard. This does not '
              'verify the physical organizer 32-GiB environment or its API route. CPU and peak RSS are sampled. '
              'A naturally completed cap-10500 run proves execution with that cap; only its actual wall time is '
              'the duration under load. Exact-CLI cancellation and isolated Linux guards are separate QA reports.', '',
              'Inspect all points rather than assuming quality increases monotonically with available time. '
              'The same task-local search may reach a different model or overfit its internal selection folds. '
              'No algorithm may be changed using the final reserved task answers.', '']
    operational_path = DIRECTORY / 'large_text_runtime.json'
    if operational_path.exists():
        operational = json.loads(operational_path.read_text())
        if operational.get('status') != 'running':
            lines += ['## Separate mixed-text operational QA', '',
                      f"Budget {operational['budget_seconds']}s, actual wall {operational['wall_seconds']:.2f}s, "
                      f"status {operational['status']}, successful fits {operational.get('fit_success')}, "
                      f"full refit {operational.get('full_refit')}, sampled RSS {operational.get('peak_memory_mb'):.2f}MiB. "
                      'This reuses supplied inputs only, with no outer answers, external scorer or quality score. '
                      'It is not another blind result, not a replacement of the original120s dummy outcomes and '
                      'not a quality point in the table above. Total process-tree CPU was not recorded by this '
                      'operational harness. Raw evidence: [large_text_runtime.json](large_text_runtime.json).', '']
    (DIRECTORY / 'budget_stress.md').write_text('\n'.join(lines))
    print(json.dumps({'completed_budget_runs': len(rows),
                      'actual_wall_seconds': sum(row.get('wall_seconds', 0) for row in rows)}))


if __name__ == '__main__':
    main()
