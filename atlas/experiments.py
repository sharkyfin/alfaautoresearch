"""Launch recipe-specific code with time/memory limits; judge stays in parent."""
from __future__ import annotations
import hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
from .budget import run_process
from .recipes import compile_recipe,recipe_hash

class Executor:
    def __init__(self,data,root,threads=8,memory_mb=24576,should_stop=lambda:False,strict=False):
        self.data=data;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.threads=threads;self.memory_mb=memory_mb;self.should_stop=should_stop;self.strict=strict
        self.counter=0
    def run(self,recipe,split,timeout,seed=42,refit=False,iterations_override=None):
        self.counter+=1;eid=f'exp_{self.counter:04d}';directory=self.root/eid;directory.mkdir()
        job={'task_path':str(self.data.inputs.task_path),'recipe':recipe,'seed':seed,'threads':self.threads,'memory_mb':self.memory_mb,'refit':refit,'iterations_override':iterations_override,'strict':self.strict}
        job_path=directory/'job.json';job_path.write_text(json.dumps(job,ensure_ascii=False))
        np.savez(directory/'split.npz',train=split.train,valid=split.valid)
        source=compile_recipe(recipe,job_path,Path(__file__).resolve().parent.parent)
        script=directory/'recipe.py';script.write_text(source)
        result=run_process([sys.executable,'-I',str(script)],directory,timeout,self.memory_mb,self.threads,self.should_stop)
        result.update(experiment_id=eid,recipe_hash=recipe_hash(recipe),source_template_hash=hashlib.sha256(source.encode()).hexdigest(),seed=seed,split_id=split.id,refit=refit,recipe=recipe)
        info_path=directory/'result.json'
        if info_path.exists():
            info=json.loads(info_path.read_text())
            if result['status']!='ok': info.pop('status',None)
            result.update(info)
        predictions=None
        if result['status']=='ok' and (directory/'predictions.npz').exists():
            with np.load(directory/'predictions.npz',allow_pickle=False) as z:predictions=(z['valid'],z['test'])
        elif result['status']=='ok':result.update(status='missing_artifact')
        result['artifact_path']=str(directory)
        # Predictions contain only inference outputs; inputs are never copied to worker directory.
        return result,predictions


def register(path,result):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
