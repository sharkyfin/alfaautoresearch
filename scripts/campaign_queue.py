"""Serialize frozen-source ML studies; keep unavailable/failed runs observable."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,required=True)
    parser.add_argument('--after',type=Path)
    args=parser.parse_args()
    state_path=args.jobs.with_suffix('.state.json')
    state={'started_at_utc':datetime.now(timezone.utc).isoformat(),'jobs':[]}
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    if args.after:
        state['waiting_for']=str(args.after);save()
        while True:
            try:
                prior=json.loads(args.after.read_text())
                if prior.get('status') in {'completed','failed'}:break
            except (OSError,ValueError):pass
            time.sleep(10)
    for job in json.loads(args.jobs.read_text()):
        report=ROOT/job['report']
        resuming=report.exists() and ('--resume' in job['command'] or
            job.get('command_mode')=='explicit_audited_final_resume')
        if report.exists() and not resuming:raise FileExistsError('Refusing to overwrite '+str(report))
        entry={'name':job['name'],'started_at_utc':datetime.now(timezone.utc).isoformat(),'command':job['command']}
        state['jobs'].append(entry);save()
        log=ROOT/job['log'] if 'log' in job else report.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a' if resuming else 'w') as output:
            result=subprocess.run([sys.executable,*job['command']],cwd=ROOT,stdout=output,stderr=output)
        entry.update(returncode=result.returncode,finished_at_utc=datetime.now(timezone.utc).isoformat(),
            status='completed' if result.returncode==0 else 'failed',report=str(report),log=str(log));save()
        print(json.dumps(entry),flush=True)
    state['status']='completed';save()


if __name__=='__main__':main()
