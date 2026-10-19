"""Build a deterministic source-only ZIP. No datasets, weights, secrets or packages."""
from __future__ import annotations
import ast,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(output=None):
    output=Path(output or ROOT/'submission.zip')
    files=[ROOT/'solution.py']+sorted((ROOT/'atlas').glob('*.py'))
    manifest=[]
    allowed_imports={'numpy','pandas','sklearn','scipy','psutil','catboost','lightgbm','xgboost'}
    import sys
    for file in files:
        source=file.read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[a.name.split('.')[0] for a in node.names]
            elif isinstance(node,ast.ImportFrom):names=[] if node.level else [(node.module or '').split('.')[0]]
            else:continue
            for name in names:
                if name not in sys.stdlib_module_names|allowed_imports|{'atlas'}:raise RuntimeError('Unapproved dependency '+name)
        for forbidden in ['site_eui','attack_cat','TabPFN','TabICL','pip install','load_pretrained']:
            if forbidden in source:raise RuntimeError('Dataset-specific or forbidden runtime token '+forbidden)
        data=file.read_bytes();manifest.append({'path':str(file.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for file in files:
            info=zipfile.ZipInfo(str(file.relative_to(ROOT)),date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,file.read_bytes())
    result={'status':'PROVISIONAL: live Qwen and Linux Docker gates remain unverified','archive':output.name,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'files':manifest}
    (ROOT/'reports/release-manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':build()
