"""Audit the current source, flat platform upload and measured live evidence."""
import argparse,hashlib,json,re,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.contract import read_inputs
from atlas.task import load
from atlas.submit import validate_frame,read_csv
from atlas.utils import code_digest
parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()

source=code_digest();test_log=(ROOT/'reports/all-tests.log').read_text();match=re.search(r'(\d+) passed(?:, (\d+) warnings)? in ([\d.]+)s',test_log)
test_state=json.loads((ROOT/'reports/test-state.json').read_text())
assert test_state['status']=='passed' and test_state['source_sha256_after']==source
assert match and 'FAILED ' not in test_log
checks=[]
for kind in ['regression','classification']:
 data=load(read_inputs(ROOT/'artifacts/data'/kind/'task.md'))
 frame=read_csv(data.inputs.output,dtype={c:'string' for c in data.spec.ids});validate_frame(data,frame)
 checks.append({'task':kind,'rows':len(frame),'columns':list(frame),'status':'valid','sha256':hashlib.sha256(data.inputs.output.read_bytes()).hexdigest()})
for script in ([] if args.audit_only else ['deadline_simulation.py','package_submission.py','verify_zip.py','report_results.py']):
 result=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],capture_output=True,text=True,timeout=120)
 (ROOT/'reports'/f'final-{script}.log').write_text(result.stdout+result.stderr)
 assert result.returncode==0,result.stdout+result.stderr
manifest=json.loads((ROOT/'reports/release-manifest.json').read_text());verified=json.loads((ROOT/'reports/zip-verification.json').read_text())
assert verified['sha256']==manifest['sha256'] and manifest['runtime_code_sha256']==source
deadline=json.loads((ROOT/'reports/deadline-simulation.json').read_text())
assert deadline['summary']['runtime_code_sha256']==source and not deadline['real_3h_soak']
with zipfile.ZipFile(ROOT/'submission.zip') as z:
 names=z.namelist();assert names[0]=='solution.py' and 'requirements.txt' in names
 assert all(Path(n).name==n and (n.endswith('.py') or n=='requirements.txt') for n in names)
 assert all(not x in n.lower() for n in names for x in ['.env','weights','dataset','__pycache__','venv'])
 for item in manifest['files']:
  data=z.read(item['path']);assert hashlib.sha256(data).hexdigest()==item['sha256']
  if item['source']!='generated bootstrap':assert data==(ROOT/item['source']).read_bytes()
blocked=[];live=[]
for path in sorted((ROOT/'reports').glob('organizer-paired-*.json')):
 r=json.loads(path.read_text())
 if r.get('status')=='complete' and r['manifest']['runtime_code_sha256']==source:live.append(r)
if not live or not any(x.get('live_fits',0)>0 for r in live for x in r['runs']):
 blocked.append('Genuine organizer-Qwen fit/evaluation and paired evidence on this source are not yet verified')
linux_path=ROOT/'reports/linux-verification.json'
linux=json.loads(linux_path.read_text()) if linux_path.exists() else {}
if linux.get('status')!='passed' or linux.get('archive_sha256')!=manifest['sha256']:
 blocked.append('Linux container execution of the current flat archive not yet verified')
result={'status':'PROVISIONAL' if blocked else 'LOCAL_RELEASE_VERIFIED','runtime_code_sha256':source,'archive_sha256':manifest['sha256'],'archive_bytes':manifest['bytes'],'tests_passed':int(match.group(1)),'warnings':int(match.group(2) or 0),'public_outputs':checks,'clean_extraction_tests':verified['tests'],'blocked':blocked,'limitations':['Local holdouts cannot establish private leaderboard quality','Scaled deadline simulation is not a real three-hour soak','Local host has 16GiB RAM; physical 32GiB conditions are not reproduced'],'platform_submitted':False}
(ROOT/'reports/final-release-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
