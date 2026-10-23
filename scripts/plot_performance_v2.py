"""Standalone research figure using only completed, matched before/after runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.report_performance_v2 import paired_effects, stats

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaigns', required=True, help='Comma-separated completed campaign JSON paths')
    parser.add_argument('--comparison', default='candidate vs baseline / deterministic')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/performance_v2/before_after.png')
    args = parser.parse_args()
    panels = []
    for name in args.campaigns.split(','):
        value = json.loads(Path(name).read_text())
        if value.get('status') != 'completed':
            raise ValueError('Plot final observations only: ' + name)
        pairs = [row for row in paired_effects(value['runs']) if row['comparison'] == args.comparison]
        if not pairs:
            raise ValueError('No matched comparison in ' + name)
        panels.append((Path(name).stem + ' / ' + value.get('stage', 'unknown'), pairs))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np
    heights = [max(1, len({p['case'] for p in pairs})) for _, pairs in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11.5, 2 + sum(heights) * .37),
                             gridspec_kw={'height_ratios': heights}, squeeze=False)
    for axis, (stage, pairs) in zip(axes[:, 0], panels):
        names = sorted({row['case'] for row in pairs})
        for i, name in enumerate(names):
            rows = sorted([row for row in pairs if row['case'] == name], key=lambda row: row['seed'])
            offset = np.linspace(-.11, .11, len(rows)) if len(rows) > 1 else [0]
            valid = [row['relative_gain'] for row in rows if 'relative_gain' in row]
            for dy, row in zip(offset, rows):
                if 'relative_gain' not in row:
                    axis.annotate(row['outcome'], (.01, i + dy), xycoords=('axes fraction', 'data'),
                                  fontsize=7, color='#666666')
                    continue
                color = {'win': '#157347', 'loss': '#ba3030', 'tie': '#777777'}[row['outcome']]
                axis.scatter(row['relative_gain'], i + dy, color=color, s=24, zorder=3)
            if len(valid) > 1:
                axis.plot([min(valid), max(valid)], [i, i], color='#aaaaaa', lw=1, zorder=1)
        result = stats(pairs)
        counts = result['outcomes']
        axis.axvline(0, color='#444444', lw=.8)
        axis.set_yticks(range(len(names)), [name.removeprefix('beyond_').replace('_', ' ') for name in names], fontsize=8)
        axis.invert_yaxis()
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis='x', alpha=.2)
        axis.set_xlabel('Relative oriented gain (positive = better); separate points = seeds')
        axis.set_title(f"{stage}: W/T/L {counts.get('win', 0)}/{counts.get('tie', 0)}/{counts.get('loss', 0)}; "
                       f"{result['distinct_datasets']} paired datasets", fontsize=10, loc='left')
        for side in ['top', 'right']:
            axis.spines[side].set_visible(False)
    fig.suptitle('Frozen-source offline benchmark — not a Boosters score', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .985))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.pdf'), bbox_inches='tight')
    print(args.output)


if __name__ == '__main__':
    main()
