"""Replay a frozen agent ZIP on original competition CSVs with an explicit test budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.kaggle_competition import prepare_competition,validate_competition_output
from scripts.kaggle_data import INPUT_FILES,snapshot_archive
from scripts.run_kaggle_full import stop_group
from scripts.run_organizer import load_environment


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_fold_metrics(summary,final_recipe):
    adjusted=summary.get('multipliers') or summary.get('ensemble')
    if adjusted:
        return [adjusted['metric_value'],*[r['metric_value'] for r in adjusted['confirmation_metrics']]]
    selected=final_recipe['components'][0]['experiment_id']
    return next(r['fold_metrics'] for r in summary['cross_validation']['candidates'] if r['experiment_id']==selected)


def run_benchmark(args):
    registry=json.loads((ROOT/'benchmarks/kaggle_competitions.json').read_text())
    spec=registry[args.competition]
    parent=ROOT/'runs';parent.mkdir(exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='branch-benchmark-',dir=parent))
    digest=snapshot_archive(args.zip,directory/'source')
    archive_sha=sha256(args.zip)
    task=directory/'inputs'
    metadata=prepare_competition(args.raw_dir,task,spec)
    runner=directory/'source'/'benchmark_entry.py'
    runner.write_text('''import os,sys,time
from pathlib import Path
START=time.monotonic()
sys.path.insert(0,str(Path(__file__).resolve().parent))
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='8'
from atlas.orchestration import run
raise SystemExit(run(sys.argv[1],started=START,budget_seconds=float(sys.argv[2]),mode='auto',threads=8,strict=True))
''')
    settings=load_environment(args.key_file)
    env={'PATH':str(Path(sys.executable).parent)+os.pathsep+os.defpath,
         'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1',**settings}
    log=directory/'pipeline.log'
    report={'status':'running','benchmark_mode':'development budget override on frozen ZIP sources',
            'competition':args.competition,'runtime_code_sha256':digest,'archive_sha256':archive_sha,
            'agent_default_overrides':{'budget_seconds':args.budget},'data':metadata,
            'run_root':str(directory),'log':str(log),'official_kaggle_score':None,
            'kaggle_submissions':0,'platform_submissions':0}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'event':'development_benchmark_start','report':str(args.report),'run_root':str(directory),
                      'budget_seconds':args.budget}),flush=True)
    started=time.monotonic();timed_out=False
    with log.open('w') as output:
        process=subprocess.Popen([sys.executable,str(runner),str(task/'task.md'),str(args.budget)],
                                 cwd=task,env=env,stdout=output,stderr=output,start_new_session=True)
        try:process.wait(timeout=args.budget+90)
        except subprocess.TimeoutExpired:timed_out=True
        finally:stop_group(process)
    report.update(wall_seconds=round(time.monotonic()-started,3),returncode=process.returncode,timed_out=timed_out)
    try:
        paths=list((task/'.atlas').glob('*/summary.json'))
        if len(paths)!=1:raise ValueError('Expected one agent summary')
        summary=json.loads(paths[0].read_text());report['agent_summary']=summary
        final=json.loads((paths[0].parent/'final_recipe.json').read_text());report['final_recipe']=final
        if digest!=summary['runtime_code_sha256'] or archive_sha!=sha256(args.zip):
            raise ValueError('Frozen runtime source or ZIP changed')
        report['inputs_unchanged']=all(sha256(task/name)==metadata['input_sha256'][name] for name in INPUT_FILES)
        if not report['inputs_unchanged']:raise ValueError('Competition inputs changed')
        history=[json.loads(line) for line in (paths[0].parent/'experiments.jsonl').read_text().splitlines()]
        report['qwen_fits']=sum(row.get('origin')=='qwen' and row['status']=='ok' for row in history)
        report['output_validation']=validate_competition_output(task/'submission.csv',task/'sample_submission.csv',spec)
        report['output_sha256']=sha256(task/'submission.csv')
        report['internal_fold_accuracy']=selected_fold_metrics(summary,final)
        report['mean_internal_accuracy']=sum(report['internal_fold_accuracy'])/len(report['internal_fold_accuracy'])
        artifacts=[log,*paths[0].parent.glob('*.json'),*paths[0].parent.glob('*.jsonl')]
        report['credential_absent_from_artifacts']=all(settings['OPENAI_API_KEY'].encode() not in p.read_bytes() for p in artifacts)
        if not report['credential_absent_from_artifacts']:raise ValueError('Credential leaked into an artifact')
        if timed_out or process.returncode or (paths[0].parent/'error.txt').exists() or not summary.get('full_refit_published'):
            report['status']='incomplete'
        else:
            delivery=directory/'submission.csv';shutil.copyfile(task/'submission.csv',delivery)
            report.update(status='completed',exported_submission=str(delivery))
    except (ValueError,KeyError,StopIteration,OSError,TypeError) as error:
        report.update(status='evaluation_error',error=str(error))
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    summary=report.get('agent_summary',{})
    lines=['# Development benchmark of branch restarts','',
           'Sources were reconstructed from the specified frozen ZIP. The official ZIP defaults were overridden for this local test.',
           f"Budget: {args.budget} seconds. Wall time: {report['wall_seconds']} seconds. Status: {report['status']}.",
           f"Experiments: {summary.get('experiments')}. Research branches: {summary.get('adaptive_search',{}).get('branches_started',1)}.",
           f"Internal fold accuracy: {report.get('internal_fold_accuracy')}; mean: {report.get('mean_internal_accuracy')}.",
           f"Qwen requests: {summary.get('qwen_calls')}; completed proposals: {report.get('qwen_fits')}.",
           f"Full-data refit: {summary.get('full_refit_published')}.",'',
           'These internal validation scores are not Kaggle leaderboard scores. No submission was made.',
           'Export: `'+report.get('exported_submission','')+'`.']
    args.report.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key:report[key] for key in ['status','wall_seconds','mean_internal_accuracy','exported_submission','error'] if key in report}),flush=True)
    return 0 if report['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition',default='spaceship_titanic')
    parser.add_argument('--zip',type=Path,default=ROOT/'submission.zip')
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--budget',type=float,default=360)
    parser.add_argument('--key-file',type=Path,default=ROOT/'.env.organizer.local')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/branch-benchmark.json')
    args=parser.parse_args()
    if not 30<=args.budget<=10500:parser.error('Benchmark budget must be between 30 and 10500 seconds')
    return run_benchmark(args)


if __name__=='__main__':raise SystemExit(main())
