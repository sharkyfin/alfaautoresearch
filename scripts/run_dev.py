"""Development-only controls; official entrypoint remains the baseline CLI."""
import argparse,os,sys,time
from pathlib import Path
START=time.monotonic()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('task',nargs='?',default='task.md');p.add_argument('--budget',type=float,default=60);p.add_argument('--mode',choices=['deterministic','mock-agent','live-agent'],default='deterministic');p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=8);p.add_argument('--memory-mb',type=int,default=24576);p.add_argument('--max-experiments',type=int,default=30);p.add_argument('--run-dir');p.add_argument('--allow-version-drift',action='store_true');a=p.parse_args()
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[v]=str(a.threads)
from atlas.orchestration import run
raise SystemExit(run(a.task,started=START,budget_seconds=a.budget,mode=a.mode,seed=a.seed,threads=a.threads,memory_mb=a.memory_mb,max_experiments=a.max_experiments,run_dir=a.run_dir,strict=not a.allow_version_drift))
