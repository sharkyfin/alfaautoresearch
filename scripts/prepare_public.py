"""Preserve the official task's relative-path layout after downloading files."""
import re,shutil
from pathlib import Path
for kind in ['classification','regression']:
    root=Path('artifacts/data')/kind
    task=root/'task.md'
    if not task.exists(): continue
    for filename in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv']:
        matches=re.findall(r'`([^`\r\n]*'+re.escape(filename)+r')`',task.read_text())
        if not matches:continue
        dest=root/matches[0]; src=root/filename
        if src.exists() and src!=dest:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(src,dest)
        print(dest,dest.exists())
