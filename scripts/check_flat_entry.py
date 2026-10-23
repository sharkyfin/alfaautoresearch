"""Bounded integration of the real flat bootstrap; not a default 3-hour run."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scripts.synthetic import create

HOOK='''import importlib.abc,importlib.machinery,sys
class Bound(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!='atlas.orchestration':return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  original=spec.loader
  class Loader(importlib.abc.Loader):
   def create_module(self,spec):return None
   def exec_module(self,module):
    original.exec_module(module)
    run=module.run
    def bounded(task_path,started=None,**kwargs):
     return run(task_path,started=started,budget_seconds=90,mode='deterministic',threads=2,memory_mb=2048,max_experiments=2)
    module.run=bounded
  spec.loader=Loader();return spec
sys.meta_path.insert(0,Bound())
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--report',required=True,type=Path);a=p.parse_args()
 with tempfile.TemporaryDirectory(prefix='atlas-flat-gate-') as temp:
  root=Path(temp);task=create(root/'upload','iid',n=160);upload=task.parent
  with zipfile.ZipFile(a.archive) as z:z.extractall(upload)
  hook=root/'hook';hook.mkdir();(hook/'sitecustomize.py').write_text(HOOK)
  env={'PATH':os.defpath,'PYTHONPATH':str(hook),'PYTHONIOENCODING':'utf-8'}
  result=subprocess.run([sys.executable,'solution.py','task.md'],cwd=upload,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=110)
  a.report.with_suffix('.log').write_text(result.stdout)
  assert result.returncode==0,result.stdout[-3000:]
  summary=json.loads(next((upload/'.atlas').glob('*/summary.json')).read_text())
  output=pd.read_csv(upload/'submission.csv',dtype={'id':str});sample=pd.read_csv(upload/'sample_submission.csv',dtype={'id':str})
  assert output.id.equals(sample.id) and list(output)==list(sample) and np.isfinite(output.response).all()
  assert summary['full_refit_published'] and summary['validation_guards']['sealed_before_search']
  # Test the actual bootstrap's fail-closed checksum guard too.
  source=upload/'atlas_models.py';source.write_bytes(source.read_bytes()+b'\n# tamper probe\n')
  tamper=subprocess.run([sys.executable,'solution.py','task.md'],cwd=upload,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
  assert tamper.returncode!=0 and 'Source integrity check failed' in tamper.stdout
  report={'status':'passed','archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),
   'command':'python3 solution.py task.md','scope':'actual flat bootstrap with development import hook supplying 90s/2 experiments; no 3-hour/default-budget execution claim',
   'full_refit':True,'output_schema_ids_finite':True,'tampered_source_rejected':True,
   'runtime_code_sha256':summary['runtime_code_sha256'],'rows':len(output),'budget_seconds':summary['budget_seconds'],'experiments':summary['experiments']}
  a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
