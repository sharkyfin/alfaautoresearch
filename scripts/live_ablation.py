"""Frozen, sequential, equal-budget deterministic versus live free-LLM comparison."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.utils import code_digest
from atlas.contract import read_inputs
from atlas.task import load
from atlas.submit import read_csv,validate_frame
from scripts.synthetic import create
from scripts.openrouter_client import BASE_URL,MODEL,load_key


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    os.replace(temporary,path)


def quota(key):
    import urllib.request
    request=urllib.request.Request(BASE_URL+'/key',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(request,timeout=20) as response:data=json.load(response)['data']
    return {'free_model_daily_requests':data.get('free_model_daily_requests'),'reported_usage_usd':data.get('usage')}


def selected(summary):
    return (summary.get('ensemble') or summary.get('multipliers') or {}).get('metric_value',summary.get('champion_metric'))


def pairs(records,live_variant='openrouter'):
    grouped={}
    for row in records:grouped.setdefault((row['task'],row['seed']),{})[row['variant']]=row
    result=[]
    for (task,seed),pair in grouped.items():
        if not {'deterministic',live_variant}<=set(pair):continue
        d,l=pair['deterministic'],pair[live_variant]
        if d['status']!='ok' or l['status']!='ok':
            result.append({'task':task,'seed':seed,'status':'incomplete_pair'});continue
        direction=d['metric_direction'];dv,lv=d['selected_metric'],l['selected_metric']
        assert d['validation']==l['validation'] and d['input_sha256']==l['input_sha256']
        difference=direction*(lv-dv)
        result.append({'task':task,'seed':seed,'status':'compared','metric':d['metric'],'deterministic':dv,live_variant:lv,'utility_difference':difference,'relative_improvement':difference/max(abs(dv),1e-12),'deterministic_seconds':d['elapsed_seconds'],'live_seconds':l['elapsed_seconds'],'live_calls':l['researcher_calls'],'live_fits':l['live_fits'],'live_accepted':l['live_accepted']})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='openrouter-paired-v1')
    parser.add_argument('--provider',choices=['openrouter','organizer'],default='openrouter')
    parser.add_argument('--omit-model',action='store_true')
    parser.add_argument('--budget',type=float,default=90)
    parser.add_argument('--seeds',default='17,43')
    parser.add_argument('--tasks',default='iid,skew,integer_labels,multiclass_imbalance')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--rows',type=int,default=600)
    parser.add_argument('--max-experiments',type=int,default=12)
    parser.add_argument('--max-calls',type=int,default=2)
    parser.add_argument('--key-file',type=Path)
    args=parser.parse_args()
    if Path(args.name).name!=args.name:parser.error('--name must be a simple directory name')
    root=ROOT/'runs'/args.name
    if root.exists():parser.error('Choose a new --name; existing campaigns are immutable')
    seeds=[int(x) for x in args.seeds.split(',')];tasks=args.tasks.split(',')
    key_path=(args.key_file or ROOT/('.env.organizer.local' if args.provider=='organizer' else '.env.openrouter.local')).resolve()
    if args.provider=='openrouter':key=load_key(key_path);before=quota(key);model=MODEL
    else:
        from scripts.run_organizer import load_environment
        settings=load_environment(key_path);model=settings.get('OPENAI_MODEL','provided-task-model');before=None
    required=len(seeds)*len(tasks)*args.max_calls
    daily=(before or {}).get('free_model_daily_requests') or {}
    if args.provider=='openrouter' and daily.get('remaining',0)<required:parser.error(f'Not enough free requests for this campaign ({required} required)')
    if required>16:parser.error('Campaign is bounded to 16 live requests; split larger experiments explicitly')
    root.mkdir(parents=True)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    source_files=[ROOT/'solution.py',*sorted((ROOT/'atlas').glob('*.py'))]
    source_files += [ROOT/'scripts'/n for n in ['run_dev.py','run_openrouter.py','openrouter_client.py','run_organizer.py','live_ablation.py']]
    hashes={}
    for file in source_files:
        target=snapshot/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
        hashes[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    runtime=code_digest();manifest={'runtime_code_sha256':runtime,'frozen_files_sha256':hashes,'provider':args.provider,'model':model,'seeds':seeds,'tasks':tasks,'rows':args.rows,'budget_seconds':args.budget,'threads':args.threads,'max_experiments':args.max_experiments,'max_calls_per_live_run':args.max_calls,'max_live_calls':required,'quota_before':before,'pid':os.getpid(),'started_at_unix':time.time(),'evaluation':'Original task metric, fixed common holdout; no test labels used','mode_order':'Alternating first variant across pairs; one process at a time'}
    write_json(root/'manifest.json',manifest)
    report=ROOT/'reports'/f'{args.name}.json';records=[]
    state={'status':'running','manifest':manifest,'runs':records,'pairs':[]}
    write_json(report,state)
    for seed_index,seed in enumerate(seeds):
        for task_index,name in enumerate(tasks):
            case=root/f'{name}_{seed}';task=create(case/'data',name,seed,n=args.rows)
            data=load(read_inputs(task))
            # Synthetic test targets are not an output-template oracle.
            sample=read_csv(data.inputs.sample,dtype={c:'string' for c in data.spec.ids})
            for col in sample.columns:
                if col not in data.spec.ids:sample[col]=data.spec.labels[0] if data.spec.classification else 0.
            sample.to_csv(data.inputs.sample,index=False)
            inputs_hash={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (case/'data').iterdir() if f.name!='submission.csv'}
            order=['deterministic',args.provider] if (seed_index+task_index)%2==0 else [args.provider,'deterministic']
            for variant in order:
                out=case/variant;log_path=case/f'{variant}.log'
                entry={'openrouter':'run_openrouter.py','organizer':'run_organizer.py','deterministic':'run_dev.py'}[variant]
                command=[sys.executable,str(snapshot/'scripts'/entry),str(task),'--budget',str(args.budget),'--threads',str(args.threads),'--max-experiments',str(args.max_experiments),'--seed',str(seed),'--run-dir',str(out)]
                if variant!='deterministic':
                    command+=['--key-file',str(key_path),'--max-calls',str(args.max_calls)]
                    if args.omit_model and variant=='organizer':command+=['--omit-model']
                else:command+=['--mode','deterministic']
                env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','OPENAI_API_KEY','OPENAI_MODEL','OPENAI_BASE_URL'}}
                row={'task':name,'seed':seed,'variant':variant,'input_sha256':inputs_hash,'log':str(log_path.relative_to(ROOT)),'status':'failed'}
                start=time.monotonic()
                with log_path.open('w') as log:
                    proc=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=log,start_new_session=True)
                    try:row['returncode']=proc.wait(timeout=args.budget+30)
                    except subprocess.TimeoutExpired:
                        import psutil
                        tree=psutil.Process(proc.pid).children(recursive=True)
                        proc.terminate()
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        for child in tree:
                            try:child.kill()
                            except psutil.NoSuchProcess:pass
                        row['returncode']=proc.returncode;row['failure']='controller_timeout'
                row['controller_seconds']=time.monotonic()-start
                summary_file=out/'summary.json'
                if summary_file.exists():
                    summary=json.loads(summary_file.read_text());row.update(summary)
                    assert row['runtime_code_sha256']==runtime,'Snapshot source mismatch'
                    row['metric_direction']=data.spec.metric.direction;row['selected_metric']=selected(summary)
                    registry=out/'experiments.jsonl'
                    history=[json.loads(line) for line in registry.read_text().splitlines()] if registry.exists() else []
                    live=[r for r in history if r.get('origin')==('qwen' if args.provider=='organizer' else 'openrouter')]
                    row['live_fits']=sum(r['status']=='ok' for r in live);row['live_accepted']=sum(r.get('accepted',False) for r in live)
                    usage=[e for e in summary.get('researcher_events',[]) if e['status']=='api_usage']
                    row['reported_live_cost_usd']=sum(e['cost_usd'] for e in usage) if usage and all(e.get('cost_usd') is not None for e in usage) else None
                    row['organizer_reported_cost']=sum(e['reported_cost'] for e in usage) if usage and all(e.get('reported_cost') is not None for e in usage) else None
                    frame=read_csv(data.inputs.output,dtype={c:'string' for c in data.spec.ids});validate_frame(data,frame)
                    shutil.copy2(data.inputs.output,out/'submission.csv')
                    row['output_valid']=True
                    if row['returncode']==0 and row['selected_metric'] is not None and not (out/'error.txt').exists():row['status']='ok'
                records.append(row);state['pairs']=pairs(records,args.provider);write_json(report,state)
                print(json.dumps({'completed':len(records),'total':2*len(tasks)*len(seeds),'task':name,'seed':seed,'variant':variant,'status':row['status'],'score':row.get('selected_metric'),'calls':row.get('researcher_calls'),'seconds':round(row['controller_seconds'],2)},ensure_ascii=False),flush=True)
    state.update(status='complete',quota_after=quota(key) if args.provider=='openrouter' else None,finished_at_unix=time.time())
    write_json(report,state)
    return 0 if all(r['status']=='ok' for r in records) else 1


if __name__=='__main__':raise SystemExit(main())
