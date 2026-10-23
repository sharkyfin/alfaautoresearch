"""Trusted experiment runtime; never publishes or evaluates final predictions."""
from __future__ import annotations
import json,os,resource,time,traceback,warnings
from pathlib import Path
import numpy as np
from .contract import read_inputs
from .task import load
from .models import fit_predict
from .utils import code_digest


def restrict_address_space(memory_mb):
    """Respect inherited finite limits; never attempt to raise a hard limit."""
    requested = int(memory_mb)*2**20
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    finite = [value for value in (soft, hard) if value != resource.RLIM_INFINITY]
    applied = min([requested, *finite])
    resource.setrlimit(resource.RLIMIT_AS, (applied, applied))
    return {'requested_bytes': requested, 'applied_bytes': applied,
            'inherited_soft_bytes': soft, 'inherited_hard_bytes': hard}


def execute(job_path,feature_function=None):
    job=json.loads(Path(job_path).read_text());root=Path(job_path).parent
    from .budget import watch_parent
    try:
        watch_parent(job.get('controller_pid'))
        # Linux supports a true virtual-address-space cap; macOS uses parent RSS guard.
        limits=restrict_address_space(job['memory_mb']) if os.sys.platform=='linux' else {}
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        loading_started=time.monotonic()
        data=load(read_inputs(job['task_path']),strict=job.get('strict',False),profile_data=False)
        load_seconds=time.monotonic()-loading_started
        data.spec.validation_plan=job.get('validation_metadata',{})
        split=np.load(root/'split.npz',allow_pickle=False)
        start=time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if job['recipe']['model_family']=='autogluon':
                from .challenger import fit_predict as challenger_fit
                pv,pt,info=challenger_fit(data,split['train'],split['valid'],root,
                    job['timeout_seconds'],job['seed'],job['threads'],job['memory_mb'],
                    policy=job.get('challenger_policy'),source_artifact=job.get('source_artifact'),refit=job.get('refit',False))
            else:
                pv,pt,info=fit_predict(data,job['recipe'],split['train'],split['valid'],seed=job['seed'],threads=job['threads'],refit=job.get('refit',False),iterations_override=job.get('iterations_override'),feature_function=feature_function,progress_path=root/'progress.json')
        info['resource_limits']=limits
        info['load_seconds']=load_seconds
        info['warnings']=[str(w.message)[:500] for w in caught[:20]]
        if not np.isfinite(pv).all() or not np.isfinite(pt).all():raise ValueError('Nonfinite worker predictions')
        np.savez_compressed(root/'predictions.npz',valid=pv,test=pt)
        info['fit_seconds']=time.monotonic()-start
        info['status']='ok';info['runtime_code_sha256']=code_digest()
        (root/'result.json').write_text(json.dumps(info))
    except Exception as error:
        info={'status':'error','exception_class':type(error).__name__,'message':str(error)[:1000],'traceback_excerpt':traceback.format_exc()[-6000:]}
        (root/'result.json').write_text(json.dumps(info));raise
