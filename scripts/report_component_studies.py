"""Write component decisions from completed, immutable benchmark observations."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.report_performance_v2 import paired_effects, stats

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'reports/performance_v2'


def load_completed(name):
    path = DIRECTORY / (name + '.json')
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if value.get('status') == 'completed' else None


def study(name, title, interpretation, filename):
    value = load_completed(name)
    if value is None:
        return
    pairs = paired_effects(value['runs'])
    summaries = {label: stats([row for row in pairs if row['comparison'] == label])
                 for label in sorted({row['comparison'] for row in pairs})}
    lines = ['# ' + title, '', interpretation, '',
             'These are offline development observations, not platform scores. '
             'Seeds share a dataset and are not independent datasets. '
             'Win/tie/loss tolerance is 0.1% relative.', '',
             '| Dataset | Seed | Arm | Metric | Outer score | Wall s | CPU s | Peak MiB | Full refit |',
             '|---|---:|---|---|---:|---:|---:|---:|---|']
    for row in value['runs']:
        def number(key):
            item = row.get(key)
            return '—' if item is None else f'{item:.9g}'
        lines.append(f"| {row['case']} | {row['seed']} | {row['arm']} | {row['metric']} | "
                     f"{number('external_metric')} | {number('wall_seconds')} | "
                     f"{number('cpu_seconds')} | {number('peak_memory_mb')} | {row.get('full_refit')} |")
    lines += ['', '## Paired effects', '', '```json', json.dumps(summaries, indent=2), '```', '',
              '## Frozen identities', '', '```json', json.dumps(value['versions'], indent=2), '```', '',
              f'Raw observations: `{name}.json`. Each row records input/split SHA and actual fit status.', '']
    (DIRECTORY / filename).write_text('\n'.join(lines))
    if pairs:
        fields = sorted({key for row in pairs for key in row})
        target = 'ensemble_ablation.csv' if name == 'ensemble-policies' else filename.removesuffix('.md') + '.csv'
        with (DIRECTORY / target).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(pairs)


def research_trace():
    campaigns = []
    for name in ['llm-a0-a1-a2', 'api-admission-live', 'p3i-annotation-followup',
                 'p3j-annotation-followup', 'final-lockbox-live', 'openml-restored-120']:
        value = load_completed(name)
        if value is None:
            continue
        runs = []
        for row in value['runs']:
            if row['mode'] != 'live-agent' or not row.get('run_dir'):
                continue
            path = Path(row['run_dir']) / 'experiments.jsonl'
            nodes = []
            for line in path.read_text().splitlines():
                record = json.loads(line)
                recipe = record.get('recipe', {})
                source = Path(record.get('artifact_path', '')) / 'recipe.py'
                nodes.append({
                    'experiment_id': record['experiment_id'],
                    'parent_experiment': recipe.get('parent_experiment'),
                    'origin': record.get('origin'), 'status': record['status'],
                    'family': recipe.get('model_family'), 'recipe_sha256': record.get('recipe_hash'),
                    'source_template_sha256': record.get('source_template_hash'),
                    'generated_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
                    'metric': record.get('metric_value'), 'accepted': record.get('accepted'),
                    'reason': record.get('accepted_reason'),
                    'artifact_path': record.get('artifact_path')})
            runs.append({'case': row['case'], 'seed': row['seed'], 'source_sha': row['source_sha'],
                         'qwen_calls': row['qwen_calls'], 'qwen_fits': row['qwen_fits'],
                         'qwen_accepted': row['qwen_accepted'], 'outer_metric': row['external_metric'],
                         'terminal_status': row.get('terminal_status'), 'nodes': nodes})
        campaigns.append({'campaign': name, 'status': value['status'], 'runs': runs})
    output = {'schema_version': 2, 'note': 'Executed development histories; internal acceptance does not imply outer improvement.',
              'campaigns': campaigns}
    (DIRECTORY / 'research_trace.json').write_text(json.dumps(output, indent=2) + '\n')


def main():
    study('native-target-encoding', 'Native categories versus cross-fitted target encoding',
          'Fixed LightGBM recipes differ only in native/target encoding; the third arm is a fixed native CatBoost '
          'reference, not an isolated encoding intervention. Three datasets, two seeds, 120-second cap including '
          'controls and full refit. Target encoding has no positive paired outer effect here. It remains an '
          'explicit optional recipe capability and is not automatically scheduled or the default. '
          'CatBoost is better on the insurance subset and worse on California/Adult for these particular recipes; '
          'this does not justify a universal CatBoost replacement.', 'native_target_encoding_study.md')
    study('seed-bagging', 'Two-seed stochastic LightGBM ablation',
          'The same stochastic LightGBM recipe uses one versus two independently seeded sequential fits on the '
          'same legal folds. Two datasets, two outer seeds, 90-second cap. Three wins and one loss: the Concrete '
          'seed-53 regression is retained in the table. Keep a budget-gated, task-validated candidate; do not force '
          'bagging on every task. This fixed-recipe study does not establish the effect of the whole scheduler.',
          'seed_bagging_study.md')
    study('ensemble-policies', 'Model averaging and fold averaging ablation',
          'The frozen source, inputs, seed and 120-second cap are shared. Arms change only ensemble selection '
          '(none, pairwise weighted, greedy up to three components), or greedy selection plus fold-average '
          'finalization. Confirmation rejects a proposed blend unless it beats every single candidate on every '
          'shared control. Fold averaging deliberately does not publish a full-data single-model refit. '
          'Compare recorded wall time, selected ensemble and output SHA; equal scores do not prove a useful blend. '
          'Wall-clock admission can produce different completed candidate pools even with the same model seed. '
          'A score difference when both ensemble fields are null is not evidence of a causal blending gain. '
          'For example, Dry Beans seed 42 completed an additional CatBoost confirmation in the greedy arm; '
          'the selected single CatBoost explains that observation.',
          'ensemble_study.md')
    value = load_completed('ensemble-policies')
    if value is not None:
        evidence = []
        for row in value['runs']:
            pool = (row.get('cross_validation') or {}).get('candidates', [])
            evidence.append({'case': row['case'], 'seed': row['seed'], 'arm': row['arm'],
                             'ensemble_accepted': bool(row.get('ensemble')),
                             'ensemble': row.get('ensemble'), 'output_sha256': row.get('output_sha256'),
                             'confirmed_candidate_pool_sha256': hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest(),
                             'confirmed_candidates': len(pool), 'champion_family': row.get('champion_family'),
                             'fold_bagged_done': row.get('fold_bagged_done')})
        (DIRECTORY / 'ensemble_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    research_trace()


if __name__ == '__main__':
    main()
