"""An exiting child is reaped; a real signal-permission failure is not hidden."""
import os
import signal
import subprocess
import sys

import pytest

from atlas import budget


def test_stop_reaps_child_that_exits_during_group_signal(monkeypatch):
    process=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)'],start_new_session=True)
    real_kill=os.kill
    def exiting_group(pid,sig):
        real_kill(pid,signal.SIGKILL)
        raise PermissionError('Sandbox EPERM for exiting process group')
    monkeypatch.setattr(budget.os,'killpg',exiting_group)
    budget.stop_process(process)
    assert process.poll() is not None


def test_stop_preserves_permission_error_for_living_child(monkeypatch):
    process=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)'],start_new_session=True)
    real_killpg=os.killpg
    def denied_group(pid,sig):raise PermissionError('Owned child still alive')
    monkeypatch.setattr(budget.os,'killpg',denied_group)
    try:
        with pytest.raises(PermissionError,match='still alive'):
            budget.stop_process(process)
        assert process.poll() is None
    finally:
        real_killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)


def test_stop_reaps_exit_race_during_sigkill_escalation(monkeypatch):
    process=subprocess.Popen([sys.executable,'-I','-c',
        'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'],
        start_new_session=True,stdout=subprocess.PIPE,text=True)
    assert process.stdout.readline().strip()=='ready'
    real_killpg=os.killpg
    def racing_group(pid,sig):
        real_killpg(pid,sig)
        if sig==signal.SIGKILL:raise PermissionError('Group exited during escalation')
    monkeypatch.setattr(budget.os,'killpg',racing_group)
    try:
        budget.stop_process(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:real_killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
