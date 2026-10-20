"""One isolated final test log tied to an unchanged runtime source digest."""
import json,platform,re,shutil,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.utils import code_digest
before=code_digest();log=ROOT/'reports'/f'tests-{uuid.uuid4().hex[:10]}.log'
command=[sys.executable,'-m','pytest','tests','-q','--tb=short']
with log.open('w') as f:result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=f,timeout=300)
after=code_digest();text=log.read_text();match=re.search(r'(\d+) passed(?:, (\d+) warnings)? in ([\d.]+)s',text)
record={'status':'passed' if result.returncode==0 and before==after and match else 'failed','source_sha256_before':before,'source_sha256_after':after,'returncode':result.returncode,'passed':int(match.group(1)) if match else 0,'warnings':int(match.group(2) or 0) if match else 0,'seconds':float(match.group(3)) if match else None,'python':sys.version,'platform':platform.platform(),'command':command,'log':str(log.relative_to(ROOT))}
shutil.copy2(log,ROOT/'reports/all-tests.log');(ROOT/'reports/test-state.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
raise SystemExit(0 if record['status']=='passed' else 1)
