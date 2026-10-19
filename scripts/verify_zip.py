"""Extract into an empty directory and exercise the root entrypoint and CLI."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.synthetic import create
p=argparse.ArgumentParser();p.add_argument('--zip',default=str(ROOT/'submission.zip'));a=p.parse_args()
records=[]
with tempfile.TemporaryDirectory(prefix='atlas-zip-') as temp:
    root=Path(temp)
    with zipfile.ZipFile(a.zip) as z:
        assert 'solution.py' in z.namelist()
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(root)
    # Run the unchanged official entrypoint, then terminate it after a model checkpoint.
    # This verifies the 175-minute default path without waiting three hours.
    for scenario in ['iid','integer_labels']:
        task=create(root/scenario,scenario,n=300)
        env={k:v for k,v in os.environ.items() if k not in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','PYTHONPATH']}
        logfile=root/f'{scenario}.log'
        with logfile.open('w') as log:
            proc=subprocess.Popen([sys.executable,str(root/'solution.py'),str(task)],cwd=root,env=env,stdout=log,stderr=log)
            deadline=time.monotonic()+50
            try:
                while time.monotonic()<deadline:
                    text=logfile.read_text()
                    if '"accepted": true' in text or proc.poll() is not None:break
                    time.sleep(.1)
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
                records.append({'scenario':scenario,'status':'passed','accepted_checkpoint':'"accepted": true' in logfile.read_text(),'returncode':proc.returncode})
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
result={'python':sys.version,'sha256':hashlib.sha256(Path(a.zip).read_bytes()).hexdigest(),'tests':records,'environment':'macOS; official CLI; no organizer API credentials; SIGTERM after first fitted checkpoint'}
(ROOT/'reports/zip-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
