"""Simulate Boosters' root-file copy filter and exact CLI in a fresh task cwd."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.synthetic import create
from atlas.utils import code_digest


def verify(archive):
    records=[]
    with tempfile.TemporaryDirectory(prefix='atlas-zip-') as temp:
        extracted=Path(temp)/'upload';extracted.mkdir()
        with zipfile.ZipFile(archive) as z:
            assert 'solution.py' in z.namelist() and 'requirements.txt' in z.namelist()
            assert len(z.namelist())==len(set(z.namelist()))
            assert all(Path(n).name==n and (n.endswith('.py') or n=='requirements.txt') for n in z.namelist())
            z.extractall(extracted)
        assert (extracted/'requirements.txt').read_bytes()==(ROOT/'artifacts/official/requirements-folder.txt').read_bytes()
        for scenario in ['iid','integer_labels']:
            task=create(Path(temp)/scenario,scenario,n=300)
            # Reproduce only the files the platform actually copies into the task.
            import shutil
            for source in extracted.iterdir():
                if source.is_file() and (source.suffix=='.py' or source.name=='requirements.txt'):
                    shutil.copy2(source,task.parent/source.name)
            env={k:v for k,v in os.environ.items() if k not in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','PYTHONPATH']}
            logfile=Path(temp)/f'{scenario}.log'
            with logfile.open('w') as log:
                proc=subprocess.Popen([sys.executable,'solution.py','task.md'],cwd=task.parent,env=env,stdout=log,stderr=log)
                deadline=time.monotonic()+50
                try:
                    while time.monotonic()<deadline:
                        text=logfile.read_text()
                        if '"accepted": true' in text or proc.poll() is not None:break
                        time.sleep(.1)
                    assert '"accepted": true' in logfile.read_text(),logfile.read_text()
                    assert (task.parent/'submission.csv').exists(),logfile.read_text()
                    proc.terminate();proc.wait(timeout=8)
                    assert proc.returncode==0,logfile.read_text()
                    import pandas as pd,numpy as np
                    result=pd.read_csv(task.parent/'submission.csv',dtype={'id':'string'})
                    test=pd.read_csv(task.parent/'test.csv',dtype={'id':'string'})
                    assert list(result.columns)==['id','response'] and len(result)==len(test) and result['id'].equals(test['id'])
                    assert result.notna().all().all()
                    if scenario=='iid':assert np.isfinite(result.response).all()
                    else:assert set(result.response)<=set(pd.read_csv(task.parent/'train.csv').response)
                    summaries=list((task.parent/'.atlas').glob('*/summary.json'));assert len(summaries)==1
                    summary=json.loads(summaries[0].read_text())
                    assert summary['runtime_code_sha256']==code_digest()
                    assert summary['budget_seconds']==900,summary['budget_seconds']
                    assert summary['cancelled'] and not (summaries[0].parent/'error.txt').exists()
                    records.append({'scenario':scenario,'status':'passed','accepted_checkpoint':True,'returncode':proc.returncode,'runtime_code_sha256':summary['runtime_code_sha256'],'default_budget_seconds':summary['budget_seconds']})
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
    return {'python':sys.version,'sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),'tests':records,'environment':'macOS; root-only platform copy; python solution.py task.md; no API credentials; SIGTERM after fitted checkpoint'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--zip',default=str(ROOT/'submission.zip'));args=parser.parse_args()
    result=verify(args.zip)
    (ROOT/'reports/zip-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
