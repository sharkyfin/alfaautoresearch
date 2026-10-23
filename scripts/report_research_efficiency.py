"""Summarize executed efficiency studies; never invent or replace observations."""
from __future__ import annotations
import csv,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/research_efficiency_20261023'
NAMES=['fixed-mixed','fixed-text-equivalence','paired-search','paired-qwen',
       'fixed-mixed-reverse','fixed-text-reverse']

def main():
    reports={name:json.loads((REPORT/(name+'.json')).read_text()) for name in NAMES if (REPORT/(name+'.json')).exists()}
    result={'status':'completed' if len(reports)==len(NAMES) and all(d['status']=='completed' for d in reports.values()) else 'running',
            'campaigns':{},'limitations':['Existing development sets; not a new blind lockbox',
                '120-second caps do not establish three-hour quality',
                'Qwen ablation changes only atlas/agent.py; live replies remain stochastic',
                'Internal hypothesis acceptance is not an external quality win',
                'No leaderboard submission or estimated leaderboard score']}
    rows=[];lines=['# Реальные результаты сравнений','',
        'Два seed42/53, одинаковые splits,120с,8threads,8192MiB. Метрика/scorer не менялись.',
        'Это повторно используемые development-наборы, не новый blind lockbox.','',
        '| Исследование | Задача | Seed | Метрика | До | После | Gain,% | Секунды до/после |',
        '|---|---|---:|---|---:|---:|---:|---|']
    cost={}
    for name,report in reports.items():
        summary=report['summary'];byarm={arm:[r for r in report['runs'] if r['arm']==arm] for arm in ['baseline','candidate']}
        detail={'status':report['status'],'observed_rows':len(report['runs']),**summary,
            'arms':{arm:{'trained':sum(r['status']=='trained_success' for r in rr),
                'fits':sum(r.get('fit_success',0) for r in rr),
                'full_refits':sum(bool(r.get('full_refit')) for r in rr),
                'qwen_calls':sum(r.get('qwen_calls',0) for r in rr),
                'qwen_tokens':sum(r.get('qwen_tokens',0) for r in rr),
                'qwen_fits':sum(r.get('qwen_fits',0) for r in rr),
                'qwen_internal_accepted':sum(r.get('qwen_accepted',0) for r in rr),
                'wall_seconds':sum(r.get('wall_seconds',0) for r in rr),
                'cpu_seconds':sum(r.get('cpu_seconds',0) for r in rr),
                'peak_memory_mb':max((r.get('peak_memory_mb',0) for r in rr),default=0),
                'failure_statuses':[r['status'] for r in rr if r['status']!='trained_success']}
                for arm,rr in byarm.items()}}
        for pair in summary['pairs']:
            case,seed=pair['case'],pair['seed']
            before=next(r for r in byarm['baseline'] if r['case']==case and r['seed']==seed)
            after=next(r for r in byarm['candidate'] if r['case']==case and r['seed']==seed)
            assert before['input_sha256']==after['input_sha256'] and before['split_sha256']==after['split_sha256']
            row={'campaign':name,'case':case,'seed':seed,'metric':before['metric'],
                 'before':before.get('external_metric'),'after':after.get('external_metric'),
                 'relative_gain':pair.get('relative_gain'),'outcome':pair['outcome'],
                 'before_status':before['status'],'after_status':after['status'],
                 'before_seconds':before.get('wall_seconds'),'after_seconds':after.get('wall_seconds'),
                 'before_fits':before.get('fit_success'),'after_fits':after.get('fit_success'),
                 'before_memory_mb':before.get('peak_memory_mb'),'after_memory_mb':after.get('peak_memory_mb'),
                 'output_byte_identical':before.get('output_sha256')==after.get('output_sha256'),
                 'before_source_sha':before['source_sha'],'after_source_sha':after['source_sha']}
            rows.append(row)
            if pair.get('relative_gain') is not None:
                lines.append(f'| {name} | {case} | {seed} | {before["metric"]} | {row["before"]:.7g} | {row["after"]:.7g} | {100*row["relative_gain"]:+.3f} | {row["before_seconds"]:.2f}/{row["after_seconds"]:.2f} |')
            else:lines.append(f'| {name} | {case} | {seed} | {before["metric"]} | — | — | — | {before["status"]}/{after["status"]} |')
            if name.startswith('fixed') and pair.get('relative_gain') is not None:
                cost.setdefault((case,seed),[]).append(row)
        result['campaigns'][name]=detail
    balanced=[]
    for key,rr in cost.items():
        if len(rr)<2:continue
        old=statistics.mean(r['before_seconds'] for r in rr);new=statistics.mean(r['after_seconds'] for r in rr)
        balanced.append({'case':key[0],'seed':key[1],'repetitions':len(rr),
            'before_seconds_mean_both_orders':old,'after_seconds_mean_both_orders':new,
            'relative_time_saved':1-new/old,'every_submission_identical':all(r['output_byte_identical'] for r in rr)})
    result['fixed_order_counterbalanced']=balanced
    result['median_fixed_relative_time_saved']=statistics.median(r['relative_time_saved'] for r in balanced) if balanced else None
    (REPORT/'EVIDENCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if rows:
        with (REPORT/'RESULTS.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines+=['','Статус: '+result['status']+'. Незапущенные/незавершённые сравнения не считаются результатами.',
        'Все исходы, время, память, успешные fits и Qwen-статистика: [EVIDENCE.json](EVIDENCE.json).',
        'Полная таблица: [RESULTS.csv](RESULTS.csv).',
        'Baseline/candidate live-Qwen отличаются только контекстом/промптом atlas/agent.py; остальные runtime-файлы одинаковы.',
        'Fixed-recipe повторы с обратным порядком учитываются как повторные измерения тех же задач, не как новые наборы.']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'campaigns':{k:v['win_tie_loss'] for k,v in result['campaigns'].items()},'counterbalanced_pairs':len(balanced),'median_fixed_relative_time_saved':result['median_fixed_relative_time_saved']},indent=2))

if __name__=='__main__':main()
