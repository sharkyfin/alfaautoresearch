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


def described_paths(description: str, filename: str) -> list[str]:
    """Accept inline code, Markdown links and plain relative CSV paths."""
    matches=[p for p in re.findall(r'`([^`\r\n]*'+re.escape(filename)+r')`',description) if '://' not in p]
    links=re.findall(r'\]\(([^\r\n()]+)\)',description)
    matches.extend(p.strip().strip('<>') for p in links if p.strip().strip('<>').endswith(filename) and '://' not in p)
    prose=re.sub(r'`[^`\r\n]*`|\[[^\]\r\n]*\]\([^\r\n()]*\)|\b\w+://[^\s<>]+',' ',description)
    plain=re.findall(r'(?<![\w./])((?:[^\s`<>\[\](){}|,;:"\x27/]+/)*'+re.escape(filename)+r')(?![\w.])',prose)
    matches.extend(p for p in plain if '://' not in p)
    return list(dict.fromkeys(matches))

def read_inputs(task_path: str | Path) -> Inputs:
    task=Path(task_path).resolve()
    description=task.read_text(encoding='utf-8') if task.exists() else ''
    root=task.parent
    paths={}
    for name in ('train.csv','test.csv','data_dictionary.csv','sample_submission.csv'):
        matches=described_paths(description,name)
        candidates=[confined(root,m) for m in matches]
        existing=list(dict.fromkeys(p for p in candidates if p.is_file()))
        if not existing and (root/name).is_file():existing=[root/name]
        if len(existing)>1: raise ValueError(f'Ambiguous input paths: {name}')
        if not existing:
            if name=='data_dictionary.csv': paths[name]=None;continue
            raise FileNotFoundError(f'Missing official input {name}: {candidates}')
        paths[name]=existing[0]
    # Boosters always scores this exact path, even if task prose mentions other CSVs.
    output=root/'submission.csv'
    return Inputs(task,description,paths['train.csv'],paths['test.csv'],paths['data_dictionary.csv'],paths['sample_submission.csv'],output)
