"""Bounded, sequential competitive runs with immutable source and private evaluator."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];PROJECT=Path('/Users/mac/Documents/projects/alfaautoresearch')
sys.path.insert(0,str(ROOT))
from scripts.competitive_data import prepare,FILES
from scripts.kaggle_data import snapshot_archive
from scripts.benchmark_kaggle import run_one
from scripts.benchmark_suite import sandbox_command
from scripts.run_organizer import load_environment

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['initial','targeted','deep'],default='initial')
 parser.add_argument('--archives',default='baseline='+str(PROJECT/'submission_before_final_optimization_036f9036f142.zip'))
 parser.add_argument('--cases',default='aps_failure,bike_sharing_temporal');parser.add_argument('--budget',type=float,default=600)
 parser.add_argument('--phase',choices=['development','final'],default='development')
 parser.add_argument('--modes',default='live-agent,autogluon,catboost');parser.add_argument('--report',type=Path,required=True)
 args=parser.parse_args()
 if args.report.exists():raise ValueError('No automatic retries/resumption; report already exists')
 if args.stage=='initial' and args.budget>900:raise ValueError('Initial per-run cap is 900 seconds')
 state_path=PROJECT/'reports/final_optimization_20261023/state.json';state=json.loads(state_path.read_text())
 archives={k:Path(v).absolute() for k,v in (x.split('=',1) for x in args.archives.split(','))}
 modes=args.modes.split(',');cases=args.cases.split(',')
 planned=len(archives)*len(modes)*len(cases)
 if args.stage=='initial' and state['initial_full_runs_used']+planned>6:raise ValueError('Six initial full runs maximum')
 if args.stage=='deep' and (planned!=2 or state['deep_pair_used']>=1 or not 1800<=args.budget<=2700):raise ValueError('One deep pair, 30–45 minutes per arm')
 run_parent=PROJECT/'runs';run_parent.mkdir(exist_ok=True)
 root=Path(tempfile.mkdtemp(prefix='competitive-'+args.stage+'-',dir=run_parent))
 versions={}
 for label,archive in archives.items():
  runtime=root/('runtime-'+label);digest=snapshot_archive(archive,runtime)
  versions[label]={'runtime':str(runtime),'source_sha':digest,'zip_sha':file_sha(archive),'archive':str(archive)}
 report={'status':'running','stage':args.stage,'phase':args.phase,'created_at_utc':now(),'run_root':str(root),
    'budget_seconds':args.budget,'threads':8,'memory_mb':16384,'max_qwen_calls':3,'planned_runs':planned,'versions':versions,'runs':[],
    'evaluator_sha256':{n:file_sha(ROOT/n) for n in ['scripts/competitive_data.py','scripts/competitive_compare.py','scripts/backend_worker.py','scripts/kaggle_worker.py','atlas/metrics.py']},
    'selection_policy':'Architecture changes must use code/inner evidence; final holdout is opened once after code freeze; no automatic expensive repetitions'}
 args.report.parent.mkdir(parents=True,exist_ok=True);save(args.report,report)
 settings=load_environment(PROJECT/'.env.organizer.local') if 'live-agent' in modes else {}
 for case in cases:
  prepared=root/('prepared-'+case);meta,truth=prepare(case,prepared,args.phase)
  for label,version in versions.items():
   for mode in modes:
    current=json.loads(state_path.read_text())
    if args.stage=='initial':
     if current['initial_full_runs_used']>=6:raise ValueError('Initial run quota exhausted')
     current['initial_full_runs_used']+=1;number=current['initial_full_runs_used']
    else:
     number=len(current.setdefault('targeted_runs',[]))+1
     current['targeted_runs'].append({'number':number,'stage':args.stage,'case':case,'arm':label,'mode':mode,'budget_seconds':args.budget,'started_at_utc':now()})
     if args.stage=='deep' and number and not current.get('deep_pair_registered'):current.update(deep_pair_used=1,deep_pair_registered=True)
    current.setdefault('run_ledger',[]).append({'stage':args.stage,'number':number,'case':case,'arm':label,'mode':mode,'archive_sha256':version['zip_sha'],'started_at_utc':now(),'status':'started'})
    save(state_path,current)
    trial=root/f'{case}-{label}-{mode}';trial.mkdir();task=trial/'task';task.mkdir()
    for name in FILES:shutil.copy2(prepared/name,task/name)
    prefix,isolation=sandbox_command(trial,Path(version['runtime']),allow_network=mode=='live-agent')
    profile=trial/'agent.sb'
    if profile.exists():
     rules=profile.read_text()
     q=lambda p:json.dumps(str(Path(p).resolve()))
     for path in [PROJECT/'artifacts/data',PROJECT/'reports',PROJECT/'.env.organizer.local',PROJECT/'.env.openrouter.local']:
      rules+='\n(deny file-read* (subpath '+q(path)+'))'
     allowed=[task,trial/'run',Path(version['runtime'])]
     rules+='\n(deny file-read* (require-all (subpath '+q(PROJECT/'runs')+') '+' '.join('(require-not (subpath '+q(p)+'))' for p in allowed)+'))\n'
     profile.write_text(rules)
    config=SimpleNamespace(budget=args.budget,threads=8,memory_mb=16384,max_calls=3,hard_deadline=True,
       sandbox_prefix=prefix,worker_python=str(PROJECT/'.bootstrap/performance_v2/backends/autogluon/bin/python'),
       exclude_families='',fixed_recipe=None,ensemble_strategy=None,finalization=None)
    print(json.dumps({'event':'start','stage':args.stage,'case':case,'arm':label,'mode':mode,'number':number}),flush=True)
    result=run_one(Path(version['runtime']),task/'task.md',trial/'run',mode,config,settings,meta,version['source_sha'],evaluation_truth=truth)
    result.update(arm=label,archive_sha256=version['zip_sha'],input_sha256=meta['input_sha256'],training_rows=meta['training_rows'],evaluation_rows=meta['evaluation_rows'],features=meta['features'],isolation=isolation)
    report['runs'].append(result);save(args.report,report)
    current=json.loads(state_path.read_text());current['run_ledger'][-1].update(status='finished',result_status=result['status'],external_metric=result.get('external_metric'))
    save(state_path,current)
    print(json.dumps({k:result.get(k) for k in ['case','arm','mode','status','external_metric','wall_seconds','full_refit','error']}),flush=True)
 report.update(status='completed',completed_at_utc=now());save(args.report,report);return 0

if __name__=='__main__':raise SystemExit(main())
