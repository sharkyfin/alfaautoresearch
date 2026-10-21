"""Audit a flat upload and complete the exact platform CLI on three fresh tasks.

The local HTTP proxy is a protocol stub with a fake key; it is not a real LLM.
No local organizer credentials are used by the executed upload.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def audit_static(archive):
    import re
    from atlas.utils import code_digest
    allowed={'numpy','pandas','sklearn','scipy','psutil','lightgbm','xgboost','httpx','optuna','atlas'}
    packages={'sklearn':'scikit-learn'}
    requirements=(ROOT/'artifacts/official/requirements-folder.txt').read_bytes()
    installed_in_image={match.group(1).lower() for line in requirements.decode().splitlines()
                        if (match:=re.match(r'([\w.-]+)(?:\[[^]]+\])?==',line.strip()))}
    imports=set()
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names)) and 'solution.py' in names and 'requirements.txt' in names
        assert all(Path(name).name==name and (name.endswith('.py') or name=='requirements.txt') for name in names)
        assert z.read('requirements.txt')==requirements
        for name in names:
            source=z.read(name)
            if not name.endswith('.py'):continue
            text=source.decode('utf-8')
            assert not re.search(r'\bsk-[A-Za-z0-9_-]{12,}',text),'Credential literal in '+name
            tree=ast.parse(text,filename=name)
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):modules=[alias.name.split('.')[0] for alias in node.names]
                elif isinstance(node,ast.ImportFrom):modules=[] if node.level else [(node.module or '').split('.')[0]]
                else:continue
                for module in modules:
                    assert module in sys.stdlib_module_names|allowed,(name,module)
                    if module not in sys.stdlib_module_names|{'atlas'}:imports.add(module)
            assert not any(token in text for token in ['pip install','load_pretrained','OPENROUTER_API_KEY','api.openai.com','site_eui','attack_cat'])
        assert all(packages.get(module,module) in installed_in_image for module in imports)
    assert Path(archive).stat().st_size<=100_000_000
    return {'status':'passed','archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            'archive_bytes':Path(archive).stat().st_size,'root_file_count':len(names),
            'official_requirements_unchanged':True,'third_party_imports':sorted(imports),'credential_literals_absent':True,
            'no_runtime_installer_or_alternative_llm':True,'runtime_code_sha256':code_digest()}


def complete_case(upload,root,scenario,nested=False):
    import numpy as np
    import pandas as pd
    from scripts.synthetic import create
    from atlas.utils import code_digest
    task=create(root,scenario,n=60)
    if nested:
        directory=root/'participants';directory.mkdir()
        for name in ['train.csv','test.csv','data_dictionary.csv','sample_submission.csv']:
            shutil.move(root/name,directory/name)
        task.write_text('Task: classification\nPrimary metric: Macro F1\n'+
                        '\n'.join('[data](participants/'+name+')' for name in ['train.csv','test.csv','data_dictionary.csv','sample_submission.csv'])+
                        '\nWrite submission.csv next to task.md. Historical example: `reports/predictions.csv`.\n')
    for source in upload.iterdir():
        if source.is_file() and (source.suffix=='.py' or source.name=='requirements.txt'):
            shutil.copy2(source,root/source.name)
    requests=[]
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            assert self.path=='/v1/chat/completions'
            assert self.headers.get('Authorization')=='Bearer platform-audit-not-real'
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append({'model':body['model']})
            if len(requests)>1:
                self.send_response(503);self.end_headers();return
            proposal={'model_family':'linear','params':{'C':1.} if scenario=='integer_labels' else {'alpha':.1},
                      'feature_expressions':[{'name':'audit_interaction','expression':'x0 * x1'}],
                      'statement':'Protocol audit: execute one named interaction','max_runtime_sec':20}
            payload=json.dumps({'model':'qwen/qwen3.8-27b','choices':[{'message':{'content':json.dumps(proposal)}}],
                                'usage':{'total_tokens':300}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(('127.0.0.1',0),Proxy)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    env={'PATH':str(Path(sys.executable).parent)+os.pathsep+os.defpath,
         'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1',
         'MPLCONFIGDIR':str(root/'mpl-cache'),
         'OPENAI_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1',
         'OPENAI_API_KEY':'platform-audit-not-real'}
    logfile=root.parent/(scenario+'-complete.log')
    start=time.monotonic()
    try:
        with logfile.open('w') as log:
            proc=subprocess.Popen(['python3','solution.py','task.md'],cwd=root,env=env,stdout=log,stderr=log)
            try:proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise AssertionError('Complete fixture exceeded 90 seconds; see '+str(logfile))
        assert proc.returncode==0,logfile.read_text()[-5000:]
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
    output=root/'submission.csv'
    assert output.is_file() and not (root/'reports'/'predictions.csv').exists()
    data_root=root/'participants' if nested else root
    frame=pd.read_csv(output,dtype={'id':'string'})
    sample=pd.read_csv(data_root/'sample_submission.csv',dtype={'id':'string'})
    train=pd.read_csv(data_root/'train.csv')
    assert list(frame.columns)==list(sample.columns) and len(frame)==len(sample)
    assert frame.id.equals(sample.id) and frame.notna().all().all()
    if scenario=='integer_labels':assert set(frame.response)<=set(train.response)
    else:assert np.isfinite(frame.response).all()
    summaries=list((root/'.atlas').glob('*/summary.json'));assert len(summaries)==1
    summary=json.loads(summaries[0].read_text())
    assert summary['runtime_code_sha256']==code_digest()
    assert summary['budget_seconds']==900 and not summary['cancelled']
    assert summary['mode']=='live-agent' and summary['qwen_calls']>=1 and requests
    assert requests[0]['model']=='qwen/qwen3.8-27b'
    history=[json.loads(line) for line in (summaries[0].parent/'experiments.jsonl').read_text().splitlines()]
    research=[r for r in history if r.get('origin')=='qwen' and r['status']=='ok']
    assert research and research[0]['recipe']['feature_expressions']
    assert summary['full_refit_published'] and not (summaries[0].parent/'error.txt').exists()
    if scenario=='future':
        assert summary['validation']['kind']=='time'
        assert len(summary['confirmation_plan'])==len(summary['confirmation_metrics'])==2
        split_paths=list(summaries[0].parent.glob('experiments/exp_*/split.npz'))
        windows={}
        for path in split_paths:
            with np.load(path) as split:
                if len(split['valid']):windows[tuple(split['valid'])]=split['train'].copy()
        assert len(windows)==3
        dates=pd.to_datetime(train.event_date)
        for valid,fit in windows.items():assert dates.iloc[fit].max()<dates.iloc[list(valid)].min()
        sets=[set(valid) for valid in windows]
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    jobs=[json.loads(path.read_text()) for path in summaries[0].parent.glob('experiments/exp_*/job.json')]
    assert all(job['threads']<=8 and job['memory_mb']<=24576 for job in jobs)
    return {'scenario':scenario,'status':'passed','seconds':round(time.monotonic()-start,3),
            'command':'python3 solution.py task.md','nested_markdown_inputs':nested,
            'only_platform_api_environment':True,'model_environment_variable_omitted':True,
            'default_budget_seconds':summary['budget_seconds'],'completed_without_signal':True,
            'full_refit_published':True,'output_path':'submission.csv','rows':len(frame),
            'validation_kind':summary['validation']['kind'],
            'planned_confirmation_windows':len(summary['confirmation_plan']),
            'completed_common_confirmation_windows':len(summary['confirmation_metrics']),
            'controlled_feature_trials':summary['feature_group_trials'],
            'feature_ablation_checks':summary['feature_ablation_checks'],
            'columns':list(frame.columns),'qwen_stub_requests':len(requests),'research_fits':len(research),
            'generated_code_files':len(list(summaries[0].parent.glob('experiments/exp_*/recipe.py'))),
            'successful_families':sorted({r['recipe']['model_family'] for r in history if r['status']=='ok'}),
            'worker_thread_budget':8,'worker_memory_limit_mb':24576,'runtime_code_sha256':code_digest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,default=ROOT/'submission.zip')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/platform-compliance.json')
    args=parser.parse_args()
    static=audit_static(args.zip)
    runs=ROOT/'runs';runs.mkdir(exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='platform-audit-',dir=runs))
    upload=directory/'upload';upload.mkdir()
    with zipfile.ZipFile(args.zip) as z:z.extractall(upload)
    cases=[complete_case(upload,directory/'regression','iid'),
           complete_case(upload,directory/'classification','integer_labels',nested=True),
           complete_case(upload,directory/'temporal','future')]
    result={'status':'LOCAL_PLATFORM_CONTRACT_PASSED','static':static,'complete_runs':cases,
            'api_check':'Local OpenAI-compatible HTTP stub, not a real LLM; no organizer key used',
            'limitations':['Organizer Docker image is unavailable locally; execution verified on Python 3.12 macOS',
                           'Supported single-target tabular tasks do not guarantee arbitrary hidden ML formulations',
                           'No leaderboard submission or public quality benchmark in this audit'],
            'platform_submitted':False,'run_dir':str(directory)}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
