"""ATLAS official entrypoint: python solution.py [task.md]."""
from __future__ import annotations
import os
import time
STARTED = time.monotonic()
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    # The controller evaluates and schedules; model subprocesses receive
    # their own full thread allowance in budget.run_process. Idle BLAS
    # pools here must not exhaust the container's PID/thread quota.
    os.environ[variable] = '1'

def main():
    import sys
    from atlas.orchestration import run
    from atlas.metrics import UnsupportedTask
    try:
        return run(sys.argv[1] if len(sys.argv)>1 else 'task.md', started=STARTED, max_experiments=480)
    except UnsupportedTask as error:
        import json
        print(json.dumps({'status':'unsupported_task','reason':str(error)},ensure_ascii=False),flush=True)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
