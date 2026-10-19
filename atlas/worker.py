"""Trusted experiment runtime; never publishes or evaluates final predictions."""
from __future__ import annotations
import json,os,resource,time,traceback
from pathlib import Path
import numpy as np
from .contract import read_inputs
from .task import load
from .models import fit_predict
from .utils import code_digest


def execute(job_path,feature_function=None):
    job=json.loads(Path(job_path).read_text());root=Path(job_path).parent
    try:
        # Linux supports a true virtual-address-space cap; macOS uses parent RSS guard.
        if os.sys.platform=='linux':resource.setrlimit(resource.RLIMIT_AS,(job['memory_mb']*2**20,job['memory_mb']*2**20))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        data=load(read_inputs(job['task_path']),strict=job.get('strict',False))
        split=np.load(root/'split.npz',allow_pickle=False)
        start=time.monotonic()
        pv,pt,info=fit_predict(data,job['recipe'],split['train'],split['valid'],seed=job['seed'],threads=job['threads'],refit=job.get('refit',False),iterations_override=job.get('iterations_override'),feature_function=feature_function,progress_path=root/'progress.json')
        if not np.isfinite(pv).all() or not np.isfinite(pt).all():raise ValueError('Nonfinite worker predictions')
        np.savez_compressed(root/'predictions.npz',valid=pv,test=pt)
        info['fit_seconds']=time.monotonic()-start
        info['status']='ok';info['runtime_code_sha256']=code_digest()
        (root/'result.json').write_text(json.dumps(info))
    except Exception as error:
        info={'status':'error','exception_class':type(error).__name__,'message':str(error)[:1000],'traceback_excerpt':traceback.format_exc()[-6000:]}
        (root/'result.json').write_text(json.dumps(info));raise
