"""Run inside the dev container, with no network, against the actual flat upload."""
from __future__ import annotations
import csv,hashlib,json,math,os,platform,shutil,subprocess,sys,time
from pathlib import Path


def rows(path):
    with path.open(newline='') as handle:return list(csv.DictReader(handle))


def main():
    upload=Path('/opt/upload');root=Path('/task');cases=[]
    expected=json.loads(Path('/verify/manifest.json').read_text())
    for item in expected['files']:
        assert hashlib.sha256((upload/item['path']).read_bytes()).hexdigest()==item['sha256']
    for scenario in ['iid','integer_labels']:
        directory=root/scenario;shutil.copytree(Path('/verify')/scenario,directory)
        for path in upload.iterdir():
            if path.is_file() and (path.suffix=='.py' or path.name=='requirements.txt'):shutil.copy2(path,directory/path.name)
        log_path=root/f'{scenario}.log'
        start=time.monotonic()
        with log_path.open('w') as log:
            process=subprocess.Popen([sys.executable,'solution.py','task.md'],cwd=directory,stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+150
                while time.monotonic()<deadline and process.poll() is None:
                    text=log_path.read_text()
                    if text.count('"event": "experiment"')>=6:break
                    time.sleep(.1)
                assert '"accepted": true' in log_path.read_text(),log_path.read_text()
                process.terminate();process.wait(timeout=12)
                assert process.returncode==0,log_path.read_text()
                prediction=rows(directory/'submission.csv');test=rows(directory/'test.csv')
                assert len(prediction)==len(test) and list(prediction[0])==['id','response']
                assert [r['id'] for r in prediction]==[r['id'] for r in test]
                assert all(r['response'] for r in prediction)
                if scenario=='iid':assert all(math.isfinite(float(r['response'])) for r in prediction)
                else:assert {r['response'] for r in prediction}<={r['response'] for r in rows(directory/'train.csv')}
                summaries=list((directory/'.atlas').glob('*/summary.json'));assert len(summaries)==1
                summary=json.loads(summaries[0].read_text())
                assert summary['runtime_code_sha256']==expected['runtime_code_sha256']
                assert summary['successful_experiments']>=6 and summary['cancelled']
                assert not (summaries[0].parent/'error.txt').exists()
                history=[json.loads(line) for line in (summaries[0].parent/'experiments.jsonl').read_text().splitlines()]
                families={r['recipe']['model_family'] for r in history if r['status']=='ok'}
                assert {'linear','lightgbm','catboost','hist','extra_trees','xgboost'}<=families
                cases.append({'scenario':scenario,'status':'passed','rows':len(prediction),'returncode':process.returncode,'seconds':time.monotonic()-start,'successful_experiments':summary['successful_experiments'],'model_families':sorted(families),'peak_worker_memory_mb':summary['peak_worker_memory_mb']})
            finally:
                if process.poll() is None:process.kill();process.wait()
    # Exercise the Linux virtual-address-space guard directly as well as fitted workers.
    limited=subprocess.run([sys.executable,'-c','import resource; resource.setrlimit(resource.RLIMIT_AS,(64*2**20,64*2**20));\ntry: bytearray(128*2**20)\nexcept MemoryError: print("blocked"); raise SystemExit(0)\nraise SystemExit(1)'],capture_output=True,text=True,timeout=5)
    assert limited.returncode==0 and 'blocked' in limited.stdout
    sys.path.insert(0,str(upload))
    from atlas_budget import run_process
    guard=root/'guards';guard.mkdir()
    memory=run_process([sys.executable,'-c','import time; x=bytearray(128*2**20); time.sleep(20)'],guard,5,memory_mb=48)
    timeout=run_process([sys.executable,'-c','import time; time.sleep(20)'],guard,.25,memory_mb=256)
    assert memory['status']=='memory_limit' and timeout['status']=='timeout'
    report={'status':'passed','archive_sha256':expected['sha256'],'runtime_code_sha256':expected['runtime_code_sha256'],'platform':platform.platform(),'python':sys.version,'tests':cases,'linux_rlimit_as':'allocation above 64MiB rejected','process_guards':{'rss':memory['status'],'wall_clock':timeout['status']},'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'network':'Docker --network=none','runtime_packages':'approved pinned subset including CatBoost/Optuna; full organizer image not available','physical_vm_memory':'8GiB; container limit 6GiB, stricter than organizer 32GiB'}
    Path('/verify/linux-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
