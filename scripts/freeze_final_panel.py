"""After exact-source gates, freeze and open the ten unseen cases exactly once.

This is local offline evaluation only. It never submits anything to Boosters.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from atlas.utils import code_digest
from scripts.benchmark_suite import EVALUATOR_FILES,file_hash,final_protocol
from scripts.verify_zip import archive_source_digest


def require(condition,message):
    if not condition:raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--gate-prefix',default='p3j')
    parser.add_argument('--freeze',type=Path,default=ROOT/'benchmarks/performance_v2/final_freeze.json')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/performance_v2/final-lockbox-live.json')
    args=parser.parse_args()
    args.report=args.report.resolve();args.freeze=args.freeze.resolve()
    require(not args.freeze.exists(),'Never overwrite a final freeze')
    require(not args.report.exists(),'Never overwrite observed final results')
    require(not args.freeze.with_name(args.freeze.stem+'.opened.json').exists(),'Lockbox already opened')
    source=archive_source_digest(args.candidate)
    require(code_digest()==source,'Working runtime differs from the selected frozen archive')
    archive_sha=file_hash(args.candidate)
    directory=ROOT/'reports/performance_v2';gate_records={}
    race=directory/'stop-race-probe-fixed.json'
    evidence=json.loads(race.read_text())
    require(evidence['counts'].get('stopped_or_reaped')==240 and
        not evidence['counts'].get('unhandled_permission_error'), 'Fixed lifecycle stress failed')
    gate_records[str(race.relative_to(ROOT))]=file_hash(race)
    for suffix in ['linux','exact-cli-matrix','denied-qwen','platform-compliance']:
        path=directory/(args.gate_prefix+'-'+suffix+'.json')
        report=json.loads(path.read_text())
        if suffix=='platform-compliance':
            require(report.get('status')=='LOCAL_PLATFORM_CONTRACT_PASSED','Exact natural CLI failed')
            identity=report['static'];require(identity.get('status')=='passed','Static compliance failed')
            require(len(report['complete_runs'])==3,'Natural CLI coverage differs')
        elif suffix=='linux':
            require(report.get('status')=='passed','Linux verification failed');identity=report
        else:
            identity={'archive_sha256':report['sha256'],'runtime_code_sha256':report['runtime_code_sha256']}
            require(all(row['status']=='passed' for row in report['tests']),'Flat CLI scenario failed')
            require(len(report['tests'])==(19 if suffix=='exact-cli-matrix' else 1),'Flat scenario coverage differs')
            if suffix=='exact-cli-matrix':
                dummy=[row for row in report['tests'] if row.get('checkpoint_kind')=='valid_dummy']
                require(all(row['scenario']=='singleton' for row in dummy),'Unexpected dummy-only scenario')
        require(identity.get('archive_sha256')==archive_sha and identity.get('runtime_code_sha256')==source,
            'Gate identity differs from selected source: '+suffix)
        gate_records[str(path.relative_to(ROOT))]=file_hash(path)
    live=directory/(args.gate_prefix+'-annotation-followup.json')
    followup=json.loads(live.read_text())
    require(followup.get('status')=='completed' and len(followup['runs'])==12,'Real Qwen follow-up incomplete')
    require(all(row['status']=='trained_success' and row['source_sha']==source for row in followup['runs']),
        'Follow-up runtime or trained output failed')
    gate_records[str(live.relative_to(ROOT))]=file_hash(live)
    # The final complete test run is serialized after all ML verification; retain earlier proofs.
    test_path=directory/(args.gate_prefix+'-final-test-gate.json')
    require(not test_path.exists(),'Do not overwrite a final test proof')
    log=directory/(args.gate_prefix+'-final-tests.log')
    with log.open('w') as output:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/check_all.py')],cwd=ROOT,stdout=output,stderr=output)
    tests=json.loads((ROOT/'reports/test-state.json').read_text())
    test_path.write_text(json.dumps(tests,indent=2)+'\n')
    require(result.returncode==0 and tests.get('status')=='passed' and tests.get('passed',0)>=301,'Full test gate failed')
    require(tests.get('source_sha256_before')==source==tests.get('source_sha256_after'),'Tests did not hold source fixed')
    gate_records[str(test_path.relative_to(ROOT))]=file_hash(test_path)
    manifest_path=ROOT/'benchmarks/manifest_v2.json'
    manifest=json.loads(manifest_path.read_text())
    cases=[row for row in manifest['cases'] if row.get('reserved_phase')=='final_lockbox']
    require(len(cases)==10,'Final case reservation changed')
    unopened=[]
    for row in cases:
        path=Path(row['cache_directory'])/'dataset.parquet'
        require(not path.exists(),'Final data were already downloaded: '+row['case'])
        unopened.append({'case':row['case'],'data_path':str(path),'data_absent':True,
            'license':row['license'],'source_revision':row.get('source_revision',row.get('revision'))})
    settings=SimpleNamespace(seeds='42,53',budget=120.0,threads=8,memory_mb=8192,max_calls=2,
        exclude_families='',exclude_by_arm='',recipe_by_arm='',blend_by_arm='',finalization_by_arm='')
    now=datetime.now(timezone.utc)
    freeze={'schema_version':2,'candidate_version':args.gate_prefix,'frozen_at_utc':now.isoformat(),
        'frozen_at_moscow':now.astimezone(ZoneInfo('Europe/Moscow')).isoformat(),
        'cases':[row['case'] for row in cases],
        'archives':{'baseline':file_hash(args.baseline),'candidate':archive_sha},
        'sources':{'baseline':archive_source_digest(args.baseline),'candidate':source},
        'protocol':final_protocol(settings,['live-agent']),
        'manifest_sha256':file_hash(manifest_path),
        'evaluator_fingerprints':{name:file_hash(ROOT/name) for name in EVALUATOR_FILES},
        'case_records_sha256':hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
        'unopened_data_checks':unopened,'gates':gate_records,
        'selection_review':'reports/performance_v2/development_regressions.md',
        'selection_review_sha256':file_hash(directory/'development_regressions.md'),
        'promotion_policy_sha256':file_hash(ROOT/'benchmarks/performance_v2/promotion_policy.json'),
        'opening_controller_sha256':file_hash(Path(__file__)),
        'selection_scope':'Chosen before final data; no runtime adaptation after opening. P3h development measurements retain P3h identity; P3i adds inert-annotation normalization; P3j only repairs the reproduced exiting-worker signal race and receives its own live and execution gates. The failed P3i live run remains a failure.',
        'evaluation':'Original curated outer repeats 0/1 (seeds42/53), parent-only truth; no platform submission.',
        'baseline_capability_exception':'Original HEAD has no MAPE; retain its unsupported row, no zero score; candidate MAPE is an unpaired capability test.',
        'stop_policy':'Report all final outcomes including losses. No reuse/tuning on these answers.'}
    args.freeze.parent.mkdir(parents=True,exist_ok=True)
    with args.freeze.open('x') as stream:json.dump(freeze,stream,indent=2);stream.write('\n')
    state_path=ROOT/'.bootstrap/performance_v2/session-state.json'
    state=json.loads(state_path.read_text());state.update(stage='P4 frozen unseen live-agent panel running',
        final_freeze=str(args.freeze.relative_to(ROOT)),final_freeze_sha256=file_hash(args.freeze),
        final_lockbox_opened=True,active_campaign=str(args.report.relative_to(ROOT)))
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    command=[sys.executable,str(ROOT/'scripts/benchmark_suite.py'),'--archives',
        'baseline='+str(args.baseline.resolve())+',candidate='+str(args.candidate.resolve()),
        '--cases','final_lockbox','--stage','final_lockbox','--freeze',str(args.freeze.resolve()),
        '--seeds','42,53','--budget','120','--threads','8','--memory-mb','8192',
        '--modes','live-agent','--max-calls','2','--report',str(args.report.resolve())]
    print(json.dumps({'status':'frozen_before_data_access','freeze':str(args.freeze),'command':command}),flush=True)
    completed=subprocess.run(command,cwd=ROOT)
    raise SystemExit(completed.returncode)


if __name__=='__main__':main()
