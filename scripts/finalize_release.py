"""Final local release audit; does not claim unavailable live/Docker gates."""
import hashlib,json,re,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.contract import read_inputs
from atlas.task import load
from atlas.submit import validate_frame,read_csv
from atlas.utils import code_digest

source=code_digest();test_log=(ROOT/'reports/all-tests.log').read_text();match=re.search(r'(\d+) passed(?:, (\d+) warnings)? in ([\d.]+)s',test_log)
test_state=json.loads((ROOT/'reports/test-state.json').read_text())
assert test_state['status']=='passed' and test_state['source_sha256_after']==source
assert match and 'FAILED ' not in test_log
checks=[]
for kind in ['regression','classification']:
 data=load(read_inputs(ROOT/'artifacts/data'/kind/'task.md'))
 frame=read_csv(data.inputs.output,dtype={c:'string' for c in data.spec.ids});validate_frame(data,frame)
 checks.append({'task':kind,'rows':len(frame),'columns':list(frame),'status':'valid','sha256':hashlib.sha256(data.inputs.output.read_bytes()).hexdigest()})
for script in ['deadline_simulation.py','package_submission.py','verify_zip.py','report_results.py']:
 result=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],capture_output=True,text=True,timeout=120)
 (ROOT/'reports'/f'final-{script}.log').write_text(result.stdout+result.stderr)
 assert result.returncode==0,result.stdout+result.stderr
manifest=json.loads((ROOT/'reports/release-manifest.json').read_text());verified=json.loads((ROOT/'reports/zip-verification.json').read_text())
assert verified['sha256']==manifest['sha256']
with zipfile.ZipFile(ROOT/'submission.zip') as z:
 names=z.namelist();assert names[0]=='solution.py'
 assert all(n=='solution.py' or n.startswith('atlas/') and n.endswith('.py') for n in names)
 assert all(not x in n.lower() for n in names for x in ['.env','weights','dataset','__pycache__','venv'])
 for n in names:assert z.read(n)==(ROOT/n).read_bytes()
result={'status':'PROVISIONAL','runtime_code_sha256':source,'archive_sha256':manifest['sha256'],'archive_bytes':manifest['bytes'],'tests_passed':int(match.group(1)),'warnings':int(match.group(2) or 0),'public_outputs':checks,'clean_extraction_tests':verified['tests'],'blocked':['Organizer Qwen settings not yet available; live hypothesis loop and live equal-budget ablation unverified','Docker unavailable; exact Linux 8CPU/32GiB verification unperformed'],'platform_submitted':False}
(ROOT/'reports/final-release-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
