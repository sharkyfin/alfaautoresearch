"""Verify the actual flat archive using the exact platform CLI in fresh tasks."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.synthetic import create


def archive_source_digest(archive):
    with zipfile.ZipFile(archive) as payload:
        digest=hashlib.sha256();digest.update(b'solution.py');digest.update(payload.read('_atlas_entry.py'))
        for filename in sorted(name for name in payload.namelist() if name.startswith('atlas_') and name.endswith('.py')):
            digest.update(filename.removeprefix('atlas_').encode());digest.update(payload.read(filename))
    return digest.hexdigest()


def add_wide_features(task):
    import numpy as np,pandas as pd
    generator=np.random.default_rng(181)
    names=['wide_'+str(i) for i in range(1500)]
    for filename in ['train.csv','test.csv']:
        path=task.parent/filename;frame=pd.read_csv(path,dtype={'id':'string'})
        extra=pd.DataFrame(generator.normal(0,1e-3,(len(frame),len(names))),columns=names)
        pd.concat([frame,extra],axis=1).to_csv(path,index=False)
    dictionary=pd.read_csv(task.parent/'data_dictionary.csv')
    extra=pd.DataFrame({'feature':names,'role':'feature','dtype':'numeric','description_ru':'Numerical feature'})
    pd.concat([dictionary,extra],ignore_index=True).to_csv(task.parent/'data_dictionary.csv',index=False)


def verify(archive,scenarios=None,abrupt=False,deny_qwen=False):
    import numpy as np,pandas as pd,psutil
    from atlas.contract import read_inputs
    from atlas.task import load,read_csv
    from atlas.submit import validate_frame
    records=[];source_sha=archive_source_digest(archive)
    with tempfile.TemporaryDirectory(prefix='atlas-zip-') as temp:
        extracted=Path(temp)/'upload';extracted.mkdir()
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            assert {'solution.py','requirements.txt'}<=set(names) and len(names)==len(set(names))
            assert all(Path(n).name==n and (n.endswith('.py') or n=='requirements.txt') for n in names)
            z.extractall(extracted)
        assert (extracted/'requirements.txt').read_bytes()==(ROOT/'artifacts/official/requirements-folder.txt').read_bytes()
        cases=[(scenario,False) for scenario in (scenarios or ['iid','integer_labels'])]
        if abrupt:cases.append(('iid',True))
        for number,(scenario,kill) in enumerate(cases):
            destination=Path(temp)/(str(number)+'-'+scenario)
            if scenario in {'official_regression','official_classification'}:
                kind=scenario.removeprefix('official_')
                original=ROOT/'artifacts/data'/kind/'task.md'
                inputs=read_inputs(original);destination.mkdir()
                task=destination/'task.md';shutil.copy2(original,task)
                for source in [inputs.train,inputs.test,inputs.dictionary,inputs.sample]:
                    target=destination/source.relative_to(original.parent);target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
            else:task=create(destination,'iid' if scenario=='wide' else scenario,n=300)
            if scenario=='wide':add_wide_features(task)
            data=load(read_inputs(task))
            for source in extracted.iterdir():
                if source.is_file() and (source.suffix=='.py' or source.name=='requirements.txt'):
                    shutil.copy2(source,task.parent/source.name)
            env={'PATH':os.defpath,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1','MPLCONFIGDIR':str(task.parent/'mpl-cache')}
            if deny_qwen:env.update(OPENAI_BASE_URL='http://127.0.0.1:9/v1',OPENAI_MODEL='denied-transport-test',OPENAI_API_KEY='disabled-test-value')
            logfile=Path(temp)/(str(number)+'.log');started=time.monotonic();descendants=[]
            with logfile.open('w') as log:
                proc=subprocess.Popen([sys.executable,'solution.py','task.md'],cwd=task.parent,env=env,stdout=log,stderr=log,start_new_session=True)
                try:
                    deadline=time.monotonic()+(600 if scenario.startswith('official_') else 100)
                    while time.monotonic()<deadline:
                        content=logfile.read_text()
                        trained='"accepted": true' in content
                        enough=(content.count('"event": "experiment"')>=3) if deny_qwen else True
                        if trained and enough:
                            if kill:
                                descendants=psutil.Process(proc.pid).children(recursive=True)
                                if not descendants and proc.poll() is None:
                                    time.sleep(.02);continue
                            break
                        if proc.poll() is not None:break
                        time.sleep(.05)
                    # The singleton fixture assigns its minority labels without
                    # feature signal. Keeping the dummy after eight successful
                    # but non-improving fits is an honest, valid outcome.
                    natural_dummy=(scenario=='singleton' and not trained and proc.poll()==0)
                    assert (trained and enough) or natural_dummy,logfile.read_text()
                    assert (task.parent/'submission.csv').exists(),logfile.read_text()
                    if kill:proc.kill()
                    elif proc.poll() is None:proc.terminate()
                    proc.wait(timeout=12)
                    assert proc.returncode==(-signal.SIGKILL if kill else 0),logfile.read_text()
                    if kill:
                        deadline=time.monotonic()+10
                        while time.monotonic()<deadline:
                            alive=[p for p in descendants if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE]
                            if not alive:break
                            time.sleep(.1)
                        assert not alive,'Orphan model workers survived controller SIGKILL'
                    result=read_csv(task.parent/'submission.csv',dtype={column:'string' for column in data.spec.ids})
                    validate_frame(data,result)
                    record={'scenario':scenario,'status':'passed','accepted_checkpoint':trained,'returncode':proc.returncode,
                        'runtime_code_sha256':source_sha,'seconds':time.monotonic()-started,'rows':len(result),
                        'termination':'natural_completion' if natural_dummy else 'SIGKILL' if kill else 'SIGTERM','default_budget_seconds':10500}
                    if not kill:
                        summaries=list((task.parent/'.atlas').glob('*/summary.json'));assert len(summaries)==1
                        summary=json.loads(summaries[0].read_text())
                        assert summary['runtime_code_sha256']==source_sha
                        assert summary['budget_seconds']==10500 and summary['cancelled']!=natural_dummy
                        assert not (summaries[0].parent/'error.txt').exists()
                        if natural_dummy:
                            assert not summary['trained_output'] and summary['checkpoint_kind']=='valid_dummy'
                            assert summary['fit_success']>=6
                            assert summary['validation']['class_support']['unvalidated_class_indices']
                            record['model_outcome']='valid_dummy_retained_after_nonimproving_successful_fits'
                            record['fit_success']=summary['fit_success']
                        else:
                            assert summary['trained_output']
                        if deny_qwen:
                            assert summary['qwen_calls']>=1 and summary['qwen_events']
                            record['denied_qwen_calls']=summary['qwen_calls']
                        record['checkpoint_kind']=summary['checkpoint_kind']
                    else:record['orphan_cleanup']='worker parent watchdog terminated isolated process group'
                    records.append(record)
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
                    for child in descendants:
                        try:
                            if child.is_running():child.kill()
                        except psutil.NoSuchProcess:pass
    return {'python':sys.version,'sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        'runtime_code_sha256':source_sha,'tests':records,'environment':'macOS; root-only platform copy; exact python solution.py task.md; no real API credentials'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--zip',default=str(ROOT/'submission.zip'))
    parser.add_argument('--scenarios',default='iid,integer_labels');parser.add_argument('--abrupt',action='store_true')
    parser.add_argument('--deny-qwen',action='store_true');parser.add_argument('--report',type=Path,default=ROOT/'reports/zip-verification.json')
    args=parser.parse_args()
    result=verify(args.zip,args.scenarios.split(','),args.abrupt,args.deny_qwen)
    args.report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
