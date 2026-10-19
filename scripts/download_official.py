"""Download public organizer files with Drive's standard confirmation form."""
from __future__ import annotations
import re, json, urllib.request, urllib.parse
from pathlib import Path
from html import unescape

def fetch(file_id: str, path: Path):
    url='https://drive.google.com/uc?'+urllib.parse.urlencode({'export':'download','id':file_id})
    with urllib.request.urlopen(url,timeout=90) as r: data=r.read()
    if data.startswith(b'<!DOCTYPE html>') and b'download-form' in data:
        s=data.decode(); fields=dict(re.findall(r'name="([^"]+)" value="([^"]*)"',s))
        action=unescape(re.search(r'action="([^"]+)"',s).group(1))
        with urllib.request.urlopen(action+'?'+urllib.parse.urlencode(fields),timeout=120) as r: data=r.read()
    if data.lstrip().startswith((b'<!DOCTYPE', b'<html')): raise RuntimeError('HTML instead of data: '+file_id)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    print(path,len(data),flush=True)

if __name__=='__main__':
    manifest=[]
    for kind in ['classification','regression']:
        s=Path(f'artifacts/official/{kind}-folder.html').read_text()
        for id_,body in re.findall(r'<tr data-selectable data-id="([^"]+)"(.*?)</tr>',s,re.S):
            name=re.search(r'<strong[^>]*>(.*?)</strong>',body,re.S)
            if name:
                name=unescape(re.sub('<[^>]+>','',name.group(1)))
                print(kind,id_,name,flush=True); manifest.append({'kind':kind,'id':id_,'name':name})
                fetch(id_,Path('artifacts/data')/kind/name)
    Path('artifacts/official/data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
