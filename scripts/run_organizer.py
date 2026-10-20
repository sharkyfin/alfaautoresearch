"""Development entrypoint using organizer settings from an ignored local file."""
from __future__ import annotations
import argparse,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def load_environment(path):
    values={}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        key,value=line.split('=',1)
        if key.strip() in {'OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL'}:
            values[key.strip()]=value.strip().strip('\"').strip("'")
    if not all(values.get(k) for k in ['OPENAI_API_KEY','OPENAI_BASE_URL']):
        raise ValueError('Organizer file must contain OPENAI_API_KEY and OPENAI_BASE_URL')
    return values


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',nargs='?',default='task.md')
    parser.add_argument('--key-file',type=Path,default=ROOT/'.env.organizer.local')
    parser.add_argument('--budget',type=float,default=120)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--memory-mb',type=int,default=24576)
    parser.add_argument('--max-experiments',type=int,default=12)
    parser.add_argument('--max-calls',type=int,default=2)
    parser.add_argument('--run-dir')
    parser.add_argument('--omit-model',action='store_true',help='Exercise the official two-variable environment')
    args=parser.parse_args()
    os.environ.update(load_environment(args.key_file))
    if args.omit_model:os.environ.pop('OPENAI_MODEL',None)
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]=str(args.threads)
    from atlas.agent import QwenClient
    from atlas.orchestration import run
    client=QwenClient('live',max_calls=args.max_calls)
    return run(args.task,started=started,budget_seconds=args.budget,mode='live-agent',seed=args.seed,threads=args.threads,memory_mb=args.memory_mb,max_experiments=args.max_experiments,run_dir=args.run_dir,research_client=client)


if __name__=='__main__':raise SystemExit(main())
