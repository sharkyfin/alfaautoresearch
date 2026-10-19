"""Small provenance helpers shared by workers and orchestration."""
import hashlib
from pathlib import Path


def code_digest():
    root=Path(__file__).resolve().parent
    files=[root.parent/'solution.py']+sorted(root.glob('*.py'))
    digest=hashlib.sha256()
    for file in files:
        digest.update(file.name.encode());digest.update(file.read_bytes())
    return digest.hexdigest()
