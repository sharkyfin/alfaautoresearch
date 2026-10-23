import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def test_worker_process_group_dies_after_controller_sigkill(tmp_path):
    root=Path(__file__).resolve().parents[1]
    child=tmp_path/'child.py'
    child.write_text('import os,sys,time\n'+f'sys.path.insert(0,{str(root)!r})\n'+
        'from atlas.budget import watch_parent\nwatch_parent(int(sys.argv[1]))\nprint("ready",flush=True)\ntime.sleep(60)\n')
    parent=tmp_path/'parent.py'
    parent.write_text('import os,subprocess,sys,time\n'+
        f'p=subprocess.Popen([sys.executable,{str(child)!r},str(os.getpid())],start_new_session=True,stdout=subprocess.PIPE,text=True)\n'+
        'assert p.stdout.readline().strip()=="ready"\nprint(p.pid,flush=True)\ntime.sleep(60)\n')
    process=subprocess.Popen([sys.executable,str(parent)],stdout=subprocess.PIPE,text=True)
    child_pid=None
    try:
        child_pid=int(process.stdout.readline().strip())
        process.kill();process.wait(timeout=3)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status()==psutil.STATUS_ZOMBIE:break
            time.sleep(.05)
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status()==psutil.STATUS_ZOMBIE
    finally:
        if process.poll() is None:process.kill();process.wait()
        if child_pid and psutil.pid_exists(child_pid):
            try:os.killpg(child_pid,signal.SIGKILL)
            except ProcessLookupError:pass
