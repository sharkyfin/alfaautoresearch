"""Read the official Markdown/cwd contract, without executing dataset content."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class Inputs:
    task_path: Path
    description: str
    train: Path
    test: Path
    dictionary: Path | None
    sample: Path
    output: Path

def confined(root: Path, path: str) -> Path:
    result=(root/path).resolve()
    if not result.is_relative_to(root.resolve()): raise ValueError(f'Path outside task workspace: {path}')
    return result

def read_inputs(task_path: str | Path) -> Inputs:
    task=Path(task_path).resolve()
    description=task.read_text(encoding='utf-8') if task.exists() else ''
    root=task.parent
    paths={}
    for name in ('train.csv','test.csv','data_dictionary.csv','sample_submission.csv'):
        matches=re.findall(r'`([^`\r\n]*'+re.escape(name)+r')`',description)
        candidates=[confined(root,m) for m in matches]
        if not candidates: candidates=[root/name]
        existing=list(dict.fromkeys(p for p in candidates if p.is_file()))
        if len(existing)>1: raise ValueError(f'Ambiguous input paths: {name}')
        if not existing:
            if name=='data_dictionary.csv': paths[name]=None;continue
            raise FileNotFoundError(f'Missing official input {name}: {candidates}')
        paths[name]=existing[0]
    # Explicitly named submission output, excluding input examples.
    matches=re.findall(r'`([^`\r\n]*(?:submission|predictions|prediction|answer)[^`\r\n]*\.csv)`',description,re.I)
    outputs=[m for m in matches if Path(m).name not in ('sample_submission.csv','train.csv','test.csv','data_dictionary.csv')]
    output=confined(root,outputs[-1] if outputs else 'submission.csv')
    return Inputs(task,description,paths['train.csv'],paths['test.csv'],paths['data_dictionary.csv'],paths['sample_submission.csv'],output)
