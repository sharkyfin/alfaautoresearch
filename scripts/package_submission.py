"""Deterministic root-only upload; own package sources materialize under /tmp."""
from __future__ import annotations
import ast,hashlib,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.utils import code_digest


def bootstrap(mapping):
    return ('''"""Official flat-upload entrypoint; reconstruct own source tree in /tmp."""
import hashlib
import runpy
import sys
import tempfile
from pathlib import Path

SOURCES='''+repr(mapping)+'''

def main():
    upload=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='atlas-runtime-',dir='/tmp') as temporary:
        runtime=Path(temporary)
        for filename,(relative,expected) in SOURCES.items():
            content=(upload/filename).read_bytes()
            if hashlib.sha256(content).hexdigest()!=expected:
                raise RuntimeError('Source integrity check failed: '+filename)
            target=runtime/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
        sys.path.insert(0,str(runtime))
        runpy.run_path(str(runtime/'solution.py'),run_name='__main__')

if __name__=='__main__':
    main()
''').encode()


def build(output=None):
    output=Path(output or ROOT/'submission.zip')
    files=[ROOT/'solution.py',*sorted((ROOT/'atlas').glob('*.py'))]
    allowed_imports={'numpy','pandas','sklearn','scipy','psutil','catboost','lightgbm','xgboost','httpx'}
    payload={};mapping={};originals={}
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
        relative=str(file.relative_to(ROOT));name='_atlas_entry.py' if file==ROOT/'solution.py' else 'atlas_'+file.name
        data=file.read_bytes();payload[name]=data;originals[name]=relative
        mapping[name]=(relative,hashlib.sha256(data).hexdigest())
    payload={'solution.py':bootstrap(mapping),**payload,
             'requirements.txt':(ROOT/'artifacts/official/requirements-folder.txt').read_bytes()}
    stage=ROOT/'artifacts/release'
    if stage.exists():shutil.rmtree(stage)
    stage.mkdir(parents=True)
    manifest=[]
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in payload.items():
            assert Path(name).name==name and (name.endswith('.py') or name=='requirements.txt')
            (stage/name).write_bytes(data)
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,data)
            manifest.append({'path':name,'source':originals.get(name,'generated bootstrap' if name=='solution.py' else 'artifacts/official/requirements-folder.txt'),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    assert output.stat().st_size<=100_000_000
    result={'status':'built; execution gates recorded separately','layout':'root Python files and official requirements.txt only','runtime_code_sha256':code_digest(),'archive':output.name,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'files':manifest}
    (ROOT/'reports/release-manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':build()
