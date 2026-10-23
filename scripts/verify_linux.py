"""Verify the flat archive in a task-local Lima VM; no host directories mounted."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.synthetic import create


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--zip',type=Path,default=ROOT/'submission.zip')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/linux-verification.json')
    args=parser.parse_args()
    lima=ROOT/'.bootstrap/lima/bin/limactl';config=ROOT/'.bootstrap/atlas-linux.yaml'
    if not lima.exists() or not config.exists():parser.error('Prepare local Lima and a no-mount VZ/Ubuntu24.04 configuration first')
    context=ROOT/'.bootstrap/linux-context'
    if context.exists():shutil.rmtree(context)
    context.mkdir()
    for name in ['Dockerfile.verify','.dockerignore','requirements-runtime.txt']:shutil.copy2(ROOT/name,context/name)
    from scripts.verify_zip import archive_source_digest
    with zipfile.ZipFile(args.zip) as payload:
        assert all(Path(name).name==name and (name.endswith('.py') or name=='requirements.txt') for name in payload.namelist())
        assert payload.read('requirements.txt')==(ROOT/'artifacts/official/requirements-folder.txt').read_bytes()
        manifest={'sha256':hashlib.sha256(args.zip.read_bytes()).hexdigest(),
            'runtime_code_sha256':archive_source_digest(args.zip),
            'files':[{'path':name,'sha256':hashlib.sha256(payload.read(name)).hexdigest()} for name in payload.namelist()]}
        payload.extractall(context/'artifacts/release')
    (context/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(ROOT/'scripts/linux_smoke.py',context/'linux_smoke.py')
    for scenario in ['iid','integer_labels']:create(context/scenario,scenario,n=300)
    assert not any(p.name.startswith('.env') for p in context.rglob('*'))
    env=os.environ.copy();env['LIMA_HOME']=str(ROOT/'.bootstrap/lima-state')
    for key in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','OPENROUTER_API_KEY']:env.pop(key,None)
    # macOS Unix sockets have a 104-byte path limit; keep the VM name short.
    name='qa';log_path=args.report.with_suffix('.vm.log');start=time.monotonic();created=False
    log_path.parent.mkdir(parents=True,exist_ok=True)
    def execute(arguments,timeout=args.timeout):
        with log_path.open('a') as log:
            log.write('\nCOMMAND '+json.dumps(arguments)+'\n');log.flush()
            result=subprocess.run([str(lima),*arguments],cwd=ROOT,env=env,stdout=log,stderr=log,timeout=timeout)
        if result.returncode:raise RuntimeError('Lima step failed; see '+str(log_path))
    log_path.write_text('Isolated Linux verification; no host directory mounts or API credentials.\n')
    try:
        created=True
        listing=subprocess.run([str(lima),'list','--json'],cwd=ROOT,env=env,capture_output=True,text=True,check=True,timeout=10)
        instances=[json.loads(line) for line in listing.stdout.splitlines() if line.strip()]
        existing=next((row for row in instances if row.get('name')==name),None)
        if existing:
            assert existing.get('cpus')==8 and existing.get('memory')==8*2**30
            assert not existing.get('config',{}).get('mounts'),'Existing VM must not expose host directories'
            execute(['start','--tty=false',name])
        else:
            execute(['start','--name='+name,'--tty=false','--mount-none',str(config)])
        execute(['copy','--backend=scp','-r',str(context),name+':/tmp/atlas-verify'])
        execute(['shell',name,'--','sh','-c','cd /tmp/atlas-verify && docker build -f Dockerfile.verify -t atlas-verify .'])
        execute(['shell',name,'--','docker','run','--rm','--cpus=8','--memory=6g','--network=none','--entrypoint=python','-v','/tmp/atlas-verify:/verify:rw','atlas-verify','/verify/linux_smoke.py'],timeout=240)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        execute(['copy','--backend=scp',name+':/tmp/atlas-verify/linux-verification.json',str(args.report.resolve())])
        report=json.loads(args.report.read_text());report['controller_seconds']=time.monotonic()-start
        assert report['archive_sha256']==manifest['sha256']
        assert report['status']=='passed'
        args.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    except Exception as error:
        report={'status':'failed','exception_class':type(error).__name__,'log':str(log_path),'controller_seconds':time.monotonic()-start}
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise
    finally:
        if created:
            try:execute(['stop','--tty=false',name],timeout=60)
            except (RuntimeError,subprocess.TimeoutExpired):pass


if __name__=='__main__':main()
