"""Monotonic budgets, bounded worker process trees, resource diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import os,signal,subprocess,time
from pathlib import Path
import psutil

@dataclass
class Budget:
    total: float=10800
    started: float=0
    def __post_init__(self):
        if not self.started:self.started=time.monotonic()
    @property
    def remaining(self):return max(0.,self.total-(time.monotonic()-self.started))
    @property
    def reserve(self):return min(300.,max(2.,self.total*.06))
    def can_run(self,estimate,refit_estimate=0,finalization_estimate=0):return self.remaining>estimate*1.3+max(self.reserve,refit_estimate*1.5,finalization_estimate)
    @property
    def search_remaining(self):return max(0.,self.remaining-self.reserve)


def stop_process(proc):
    if proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=3)


def run_process(command,cwd: Path,timeout: float,memory_mb=24576,threads=8,should_stop=lambda:False):
    # Deliberately do not forward API keys, home directories, proxy variables or PYTHONPATH.
    env={'PATH':os.defpath,'PYTHONIOENCODING':'utf-8','PYTHONHASHSEED':'0','PYTHONUNBUFFERED':'1'}
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[key]=str(threads)
    log_path=cwd/'worker.log';start=time.monotonic();peak=0.;cpu=0.;max_threads=0;reason=None
    with log_path.open('w') as log:
        proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (cwd/'worker.pid').write_text(str(proc.pid))
        try:
            while proc.poll() is None:
                if should_stop():reason='cancelled';break
                if time.monotonic()-start>timeout:reason='timeout';break
                try:
                    root=psutil.Process(proc.pid);processes=[root]+root.children(recursive=True)
                    mem=sum(p.memory_info().rss for p in processes if p.is_running())/2**20
                    peak=max(peak,mem)
                    max_threads=max(max_threads,sum(p.num_threads() for p in processes if p.is_running()))
                    cpu=max(cpu,sum(sum(p.cpu_times()[:2]) for p in processes if p.is_running()))
                    if mem>memory_mb:reason='memory_limit';break
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                time.sleep(.05)
        finally:
            if proc.poll() is None:stop_process(proc)
        rc=proc.wait()
    return {'status':reason or ('ok' if rc==0 else 'worker_error'),'returncode':rc,'runtime_sec':time.monotonic()-start,'peak_memory_mb':peak,'max_live_threads':max_threads,'cpu_seconds':cpu,'traceback_excerpt':log_path.read_text(errors='replace')[-6000:] if reason or rc else ''}
