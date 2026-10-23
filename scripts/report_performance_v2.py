"""Compile observed V2 runs, paired effects and costs without inventing scores."""
from __future__ import annotations
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LOSSES={'rmse','mse','mae','mape','rmsle','logloss'}
UNAVAILABLE={'download_blocked','missing_data','dataset_not_licensed','unsupported_metric','preparation_failed'}
SOURCE_UNAVAILABLE=UNAVAILABLE-{'unsupported_metric'}


def resource_stats(rows):
    available=[row for row in rows if row['status'] not in SOURCE_UNAVAILABLE]
    supported=[row for row in available if row['status']!='unsupported_metric']
    trained=sum(row['status']=='trained_success' for row in available)
    output={'observed_rows':len(rows),'available_rows':len(available),'supported_rows':len(supported),
            'trained_rows':trained,'status_counts':dict(Counter(row['status'] for row in rows)),
            'trained_fraction_available':trained/len(available) if available else None,
            'trained_fraction_supported':trained/len(supported) if supported else None,
            'full_refit_rows':sum(bool(row.get('full_refit')) for row in available)}
    for key in ['wall_seconds','first_trained_checkpoint_seconds','cpu_seconds','peak_memory_mb']:
        values=[row[key] for row in available if row.get(key) is not None]
        output[key+'_p50']=float(np.median(values)) if values else None
        output[key+'_p95']=float(np.quantile(values,.95)) if values else None
        output[key+'_max']=float(max(values)) if values else None
    return output


def effect(before,after,label):
    result={'comparison':label,'case':after['case'],'seed':after['seed'],'metric':after['metric'],
        'baseline_status':before['status'],'candidate_status':after['status'],
        'baseline_score':before.get('external_metric'),'candidate_score':after.get('external_metric'),
        'baseline_seconds':before.get('wall_seconds'),'candidate_seconds':after.get('wall_seconds')}
    if before['status']==after['status']=='trained_success':
        for key in ['metric','budget_seconds','split_sha256','input_sha256']:
            if key in before and key in after and before[key]!=after[key]:raise ValueError('Unmatched comparison '+key)
        direction=-1 if after['metric'] in LOSSES else 1
        old,new=before['external_metric'],after['external_metric']
        result.update(gain=direction*(new-old),relative_gain=direction*(new-old)/max(abs(old),1e-12))
        result['outcome']='win' if result['relative_gain']>.001 else 'loss' if result['relative_gain']<-.001 else 'tie'
        if old>0 and new>0:result['oriented_ratio']=new/old if direction>0 else old/new
    else:result['outcome']='unavailable' if before['status'] in UNAVAILABLE or after['status'] in UNAVAILABLE else 'execution_failure'
    return result


def paired_effects(rows):
    out=[]
    for case,seed in sorted({(row['case'],row['seed']) for row in rows}):
        group=[row for row in rows if (row['case'],row['seed'])==(case,seed)]
        by_key={(row.get('arm','reference'),row['mode']):row for row in group}
        for (arm,mode),row in by_key.items():
            if arm!='baseline' and ('baseline',mode) in by_key:
                out.append(effect(by_key[('baseline',mode)],row,arm+' vs baseline / '+mode))
            if mode!='deterministic' and (arm,'deterministic') in by_key:
                out.append(effect(by_key[(arm,'deterministic')],row,mode+' vs deterministic / '+arm))
    return out


def stats(pairs):
    values=[row['relative_gain'] for row in pairs if 'relative_gain' in row]
    grouped=defaultdict(list)
    for row in pairs:
        if 'relative_gain' in row:grouped[row['case']].append(row['relative_gain'])
    per_dataset=np.array([np.mean(value) for value in grouped.values()])
    ratios=[row['oriented_ratio'] for row in pairs if 'oriented_ratio' in row]
    result={'outcomes':dict(Counter(row['outcome'] for row in pairs)),
        'paired_observations':len(values),'distinct_datasets':len(grouped),
        'median_relative_gain':float(np.median(values)) if values else None,
        'relative_gain_25th_percentile':float(np.quantile(values,.25)) if values else None,
        'worst_quartile_mean':float(np.mean(sorted(values)[:max(1,math.ceil(len(values)/4))])) if values else None,
        'worst_single_relative_gain':min(values) if values else None,
        'geometric_oriented_ratio':float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
        'per_dataset_mean_relative_gain':{key:float(np.mean(value)) for key,value in grouped.items()}}
    if len(per_dataset)>=3:
        rng=np.random.default_rng(2026)
        samples=np.median(rng.choice(per_dataset,(2000,len(per_dataset)),replace=True),axis=1)
        result['exploratory_dataset_bootstrap_median_95_interval']=np.quantile(samples,[.025,.975]).tolist()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'reports/performance_v2')
    args=parser.parse_args();directory=args.directory;rows=[];comparisons={};campaigns=[];resources=[]
    statuses={'ok':'trained_success','constant_fallback':'fall_back_dummy','agent_error':'runtime_failed'}
    for path in sorted(directory.glob('*.json')):
        if path.stem in {'benchmark_matrix','paired_results'}:continue
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or not isinstance(value.get('runs'),list):continue
        if not all(isinstance(row,dict) and {'case','mode','status'}<=set(row) for row in value['runs']):continue
        normalized=[]
        for record in value['runs']:
            row={**record,'campaign':path.stem,'campaign_status':value.get('status')}
            row['status']=statuses.get(row.get('status'),row.get('status'))
            row.setdefault('source_sha',row.get('runtime_code_sha256',value.get('runtime_code_sha256')))
            row.setdefault('zip_sha',value.get('archive_sha256'))
            summary_path=Path(row['run_dir'])/'summary.json' if row.get('run_dir') else None
            if summary_path and not summary_path.is_absolute():summary_path=ROOT/summary_path
            if summary_path and summary_path.exists():
                try:
                    runtime_summary=json.loads(summary_path.read_text())
                    row['runtime_metric']=runtime_summary.get('metric') if isinstance(runtime_summary,dict) else None
                except (OSError,ValueError):row['runtime_metric']=None
            normalized.append(row);rows.append(row)
        pairs=paired_effects(normalized)
        for label in sorted({pair['comparison'] for pair in pairs}):
            selected=[pair for pair in pairs if pair['comparison']==label]
            comparisons[path.stem+' / '+label]={'statistics':stats(selected),'pairs':selected}
        campaigns.append({'campaign':path.stem,'status':value.get('status'),'rows':len(normalized),
            'status_counts':dict(Counter(row['status'] for row in normalized))})
        for arm,mode in sorted({(row.get('arm','reference'),row['mode']) for row in normalized}):
            selected=[row for row in normalized if row.get('arm','reference')==arm and row['mode']==mode]
            resources.append({'campaign':path.stem,'arm':arm,'mode':mode,**resource_stats(selected)})
    output={'schema_version':2,'platform_submissions':0,'campaigns':campaigns,'comparisons':comparisons,
        'statistical_note':'Quality is conditional on paired trained success. Failures and source unavailability are separate. Dataset-bootstrap intervals are exploratory; development tuning and multiple testing preclude confirmatory claims.',
        'resources':resources,'runs':rows}
    (directory/'benchmark_matrix.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    fields=['campaign','campaign_status','case','source','dataset','license','arm','mode','seed','metric','runtime_metric','external_metric','status',
        'budget_seconds','wall_seconds','first_trained_checkpoint_seconds','cpu_seconds','peak_memory_mb','fit_success',
        'control_folds_completed','full_refit','fold_bagged_done','champion_family','qwen_calls','qwen_fits','qwen_accepted',
        'qwen_tokens','llm_valid_recipes','checkpoint_kind','terminal_status','fallback_kind','output_sha256',
        'training_rows','evaluation_rows','features','n_classes','missingness','imbalance_ratio','train_class_support',
        'original_source_metric','metric_override','evaluation_unit','official_repeat','official_fold',
        'source_sha','zip_sha','split_sha256','input_sha256','raw_sha256','champion_recipe','intervention','error']
    with (directory/'benchmark_matrix.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader()
        writer.writerows({key:json.dumps(value,ensure_ascii=False,sort_keys=True) if isinstance(value,(dict,list)) else value
                          for key,value in row.items()} for row in rows)
    all_pairs=[{'campaign':label.split(' / ',1)[0],**pair}
               for label,value in comparisons.items() for pair in value['pairs']]
    if all_pairs:
        pair_fields=list(dict.fromkeys(key for pair in all_pairs for key in pair))
        with (directory/'all_paired_results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,pair_fields);writer.writeheader();writer.writerows(all_pairs)
    if resources:
        resource_fields=[key for key in resources[0] if key!='status_counts']
        with (directory/'resource_summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,resource_fields,extrasaction='ignore');writer.writeheader();writer.writerows(resources)
    lines=['# Observed V2 benchmark results','',
        'This file is generated from executed campaign artifacts. Running campaigns are incomplete. Blank metrics are not zero. No Boosters submission has been made.','',
        '| Campaign | State | Trained / observed rows | Other statuses |','|---|---|---:|---|']
    for campaign in campaigns:
        counts=campaign['status_counts'];other={key:value for key,value in counts.items() if key!='trained_success'}
        lines.append(f"| {campaign['campaign']} | {campaign['status']} | {counts.get('trained_success',0)}/{campaign['rows']} | {other} |")
    lines+=['','## Paired effects','','Positive normalized gain is better. Win/tie/loss tolerance: 0.1% relative.','',
        '| Campaign / comparison | Valid pairs | W/T/L | Median gain | 25th percentile | Worst pair |','|---|---:|---|---:|---:|---:|']
    def percent(value):return '—' if value is None else f'{100*value:+.3f}%'
    for label,value in comparisons.items():
        summary=value['statistics'];counts=summary['outcomes']
        lines.append(f"| {label} | {summary['paired_observations']} | {counts.get('win',0)}/{counts.get('tie',0)}/{counts.get('loss',0)} | {percent(summary['median_relative_gain'])} | {percent(summary['relative_gain_25th_percentile'])} | {percent(summary['worst_single_relative_gain'])} |")
    lines+=['','Full scores, statuses, costs and exact source/ZIP/split identities: `benchmark_matrix.csv` and `benchmark_matrix.json`.',
            'Every observed research comparison, including unavailable/error pairs: [all_paired_results.csv](all_paired_results.csv). '
            'Rows can share data or a reference run and are not independent experiments.',
            'Per-arm/mode latency, CPU, RSS p50/p95 and trained/available versus trained/supported fractions: `resource_summary.csv`. '
            'Unsupported metrics remain available-task capability failures; source download failures are reported separately.','']
    (directory/'benchmark_dashboard.md').write_text('\n'.join(lines))
    print(json.dumps({'campaigns':len(campaigns),'observed_rows':len(rows),'comparisons':len(comparisons)}))


if __name__=='__main__':main()
