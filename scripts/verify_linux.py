"""Verify the flat archive in a task-local Lima VM; no host directories mounted."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.synthetic import create


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timeout',type=int,default=1200);args=parser.parse_args()
    lima=ROOT/'.bootstrap/lima/bin/limactl';config=ROOT/'.bootstrap/atlas-linux.yaml'
    if not lima.exists() or not config.exists():parser.error('Prepare local Lima and a no-mount VZ/Ubuntu24.04 configuration first')
    context=ROOT/'.bootstrap/linux-context'
    if context.exists():shutil.rmtree(context)
    context.mkdir()
    for name in ['Dockerfile.verify','.dockerignore','requirements-runtime.txt']:shutil.copy2(ROOT/name,context/name)
    shutil.copytree(ROOT/'artifacts/release',context/'artifacts/release')
    shutil.copy2(ROOT/'reports/release-manifest.json',context/'manifest.json')
    shutil.copy2(ROOT/'scripts/linux_smoke.py',context/'linux_smoke.py')
    for scenario in ['iid','integer_labels']:create(context/scenario,scenario,n=300)
    assert not any(p.name.startswith('.env') for p in context.rglob('*'))
    env=os.environ.copy();env['LIMA_HOME']=str(ROOT/'.bootstrap/lima-state')
    for key in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','OPENROUTER_API_KEY']:env.pop(key,None)
    # macOS Unix sockets have a 104-byte path limit; keep the VM name short.
    name='qa';log_path=ROOT/'reports/linux-verification.log';start=time.monotonic();created=False
    def execute(arguments,timeout=args.timeout):
        with log_path.open('a') as log:
            log.write('\nCOMMAND '+json.dumps(arguments)+'\n');log.flush()
            result=subprocess.run([str(lima),*arguments],cwd=ROOT,env=env,stdout=log,stderr=log,timeout=timeout)
        if result.returncode:raise RuntimeError('Lima step failed; see reports/linux-verification.log')
    log_path.write_text('Isolated Linux verification; no host directory mounts or API credentials.\n')
    try:
        created=True
        execute(['start','--name='+name,'--tty=false','--mount-none',str(config)])
        execute(['copy','--backend=scp','-r',str(context),name+':/tmp/atlas-verify'])
        execute(['shell',name,'--','sh','-c','cd /tmp/atlas-verify && docker build -f Dockerfile.verify -t atlas-verify .'])
        execute(['shell',name,'--','docker','run','--rm','--cpus=8','--memory=6g','--network=none','--entrypoint=python','-v','/tmp/atlas-verify:/verify:rw','atlas-verify','/verify/linux_smoke.py'],timeout=240)
        execute(['copy','--backend=scp',name+':/tmp/atlas-verify/linux-verification.json',str(ROOT/'reports/linux-verification.json')])
        report=json.loads((ROOT/'reports/linux-verification.json').read_text());report['controller_seconds']=time.monotonic()-start
        manifest=json.loads((ROOT/'reports/release-manifest.json').read_text())
        assert report['archive_sha256']==manifest['sha256']
        assert report['status']=='passed'
        (ROOT/'reports/linux-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    except Exception as error:
        report={'status':'failed','exception_class':type(error).__name__,'log':'reports/linux-verification.log','controller_seconds':time.monotonic()-start}
        (ROOT/'reports/linux-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise
    finally:
        if created:
            try:execute(['stop','--tty=false',name],timeout=60)
            except (RuntimeError,subprocess.TimeoutExpired):pass


if __name__=='__main__':main()
