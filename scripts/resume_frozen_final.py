"""Explicit, audited missing-only continuation of an interrupted final panel.

The original evaluator file is NOT edited. Its exact frozen trial-loop AST is
executed after identity checks. Completed observations (including failures) are
never rerun; outer answers cannot select recipes or change the protocol.
"""
from __future__ import annotations
import ast,copy,hashlib,json,os,shutil,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='8'
from scripts import benchmark_suite as factory
from scripts.kaggle_data import snapshot_archive,INPUT_FILES
from scripts.benchmark_kaggle import run_one
from scripts.run_organizer import load_environment
from atlas.utils import code_digest


def trial_key(row):return row['case'],row['seed'],row['arm'],row['mode']

def remaining_keys(planned,rows):
    observed=[trial_key(row) for row in rows]
    if len(observed)!=len(set(observed)):raise ValueError('Duplicate completed trial keys')
    if not set(observed)<=set(planned):raise ValueError('Completed key outside frozen protocol')
    return set(planned)-set(observed)


def main():
    freeze_path=ROOT/'benchmarks/performance_v2/final_freeze.json'
    report_path=ROOT/'reports/performance_v2/final-lockbox-live.json'
    freeze=json.loads(freeze_path.read_text());report=json.loads(report_path.read_text())
    if report.get('status')!='running':raise RuntimeError('Only an interrupted running panel can continue')
    opening=freeze_path.with_name('final_freeze.opened.json')
    if json.loads(opening.read_text())['freeze_sha256']!=factory.file_hash(freeze_path):raise RuntimeError('Opening freeze differs')
    if code_digest()!=freeze['sources']['candidate']:raise RuntimeError('Selected runtime changed')
    for name,expected in freeze['evaluator_fingerprints'].items():
        if factory.file_hash(ROOT/name)!=expected:raise RuntimeError('Frozen evaluator changed: '+name)
    for name,expected in freeze['gates'].items():
        if factory.file_hash(ROOT/name)!=expected:raise RuntimeError('Frozen gate changed: '+name)
    protocol=freeze['protocol']
    args=SimpleNamespace(manifest=ROOT/'benchmarks/manifest_v2.json',report=report_path,
        budget=protocol['budget_seconds'],threads=protocol['threads'],memory_mb=protocol['memory_mb'],
        max_calls=protocol['max_qwen_calls'],seeds=','.join(map(str,protocol['seeds'])),
        exclude_families='',exclude_by_arm='',recipe_by_arm='',blend_by_arm='',finalization_by_arm='',
        stage='final_lockbox',hard_deadline=True,autogluon_python=ROOT/'.bootstrap/performance_v2/backends/autogluon/bin/python')
    modes=protocol['modes'];selected=freeze['cases']
    archives={'baseline':ROOT/'.bootstrap/performance_v2/baseline/source/submission.zip',
        'candidate':ROOT/'.bootstrap/performance_v2/p3j/source/submission.zip'}
    hashes={key:factory.file_hash(path) for key,path in archives.items()}
    factory.validate_final_freeze(freeze,selected,hashes,factory.final_protocol(args,modes),
        factory.file_hash(args.manifest),report['evaluator_fingerprints'])
    if {k:v['zip_sha'] for k,v in report['versions'].items()}!=hashes:raise RuntimeError('Report archives changed')
    planned=[(case,seed,arm,mode) for case in selected for seed in protocol['seeds'] for arm in archives for mode in modes]
    missing=remaining_keys(planned,report['runs']);preserved=copy.deepcopy(report['runs'])
    for row in preserved:
        if row['source_sha']!=freeze['sources'][row['arm']] or row['zip_sha']!=hashes[row['arm']] or row['budget_seconds']!=args.budget:
            raise RuntimeError('Completed observation identity changed')
    # Prevent accidental overlap with a still-running original evaluator.
    import psutil
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        cmd=proc.info['cmdline'] or []
        if any(str(part).endswith('scripts/benchmark_suite.py') for part in cmd) and 'final_lockbox' in cmd:
            raise RuntimeError('Original final evaluator is still running')
    audit_path=ROOT/'benchmarks/performance_v2/final_freeze.resume-01.json'
    if audit_path.exists():raise RuntimeError('This explicit resumption audit already exists; do not silently repeat')
    snapshot=ROOT/'.bootstrap/performance_v2/interrupted/final-lockbox-before-resume-01.json'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():raise RuntimeError('Never overwrite preserved partial observations')
    shutil.copy2(report_path,snapshot)
    root=Path(tempfile.mkdtemp(prefix='performance-v2-final-resume-',dir=ROOT/'runs'))
    versions={}
    for arm,path in archives.items():
        runtime=root/('runtime-'+arm);source=snapshot_archive(path,runtime)
        if source!=freeze['sources'][arm]:raise RuntimeError('Resumed runtime snapshot differs')
        versions[arm]={'runtime':str(runtime),'zip_sha':hashes[arm],'source_sha':source}
    audit={'resumed_at_utc':datetime.now(timezone.utc).isoformat(),'reason':'User requested continuation; no original evaluator/queue processes remain, partial report is running with15 completed rows.',
        'freeze_sha256':factory.file_hash(freeze_path),'opened_marker_sha256':factory.file_hash(opening),
        'preserved_snapshot':str(snapshot),'preserved_snapshot_sha256':factory.file_hash(snapshot),
        'preserved_rows':len(preserved),'missing_keys':[list(key) for key in planned if key in missing],
        'runtime_scorer_archives_protocol_unchanged':True,'resume_driver_sha256':factory.file_hash(Path(__file__)),
        'new_run_root':str(root),'new_versions':versions,
        'policy':'No completed key retried, no selection by observed scores; execute only missing keys with exact frozen trial-loop source. Preserve all interrupted artifacts.'}
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    report.setdefault('resumptions',[]).append(audit)
    settings=load_environment(ROOT/'.env.organizer.local')
    cases={r['case']:r for r in json.loads(args.manifest.read_text())['cases']}
    namespace={**vars(factory),'args':args,'root':root,'selected':selected,'modes':modes,'versions':versions,
        'report':report,'completed':set(trial_key(r) for r in preserved),'cases':cases,
        'default_exclusions':'','per_arm_exclusions':{},'policies':{},'per_arm_recipes':{},
        'INPUT_FILES':INPUT_FILES,'settings':settings,'run_one':run_one}
    source=Path(factory.__file__).read_text();tree=ast.parse(source)
    main_node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    loop=next(node for node in main_node.body if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='name' and isinstance(node.iter,ast.Name) and node.iter.id=='selected')
    factory.save_report(report_path,report)
    print(json.dumps({'event':'audited_final_resume','preserved':len(preserved),'missing':len(missing),'audit':str(audit_path)}),flush=True)
    exec(compile(ast.Module(body=[loop],type_ignores=[]),str(factory.__file__),'exec'),namespace)
    if report['runs'][:len(preserved)]!=preserved:raise RuntimeError('Completed observations were modified')
    if remaining_keys(planned,report['runs']):raise RuntimeError('Final panel still missing trials')
    report['status']='completed';factory.save_report(report_path,report)
    print(json.dumps({'event':'final_completed','rows':len(report['runs'])}),flush=True)

if __name__=='__main__':main()
