"""Version-specific experiment reliability/resource counters from actual registries."""
import collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];reports=[]
for name in ['public-campaign','public-final']:
 path=ROOT/'reports'/f'{name}.json'
 if not path.exists():continue
 for run in json.loads(path.read_text()):
  history=[json.loads(line) for line in (ROOT/'runs'/name/f"{run['task']}_{run['seed']}"/'experiments.jsonl').read_text().splitlines()]
  counts=collections.Counter(x['status'] for x in history)
  reports.append({'campaign':name,'task':run['task'],'seed':run['seed'],'runtime_code_sha256':run['runtime_code_sha256'],'experiments':len(history),'statuses':dict(counts),'observed_memory_limit_exits':counts.get('memory_limit',0),'worker_timeout_rate':counts.get('timeout',0)/len(history) if history else None,'worker_cpu_seconds':sum(x.get('cpu_seconds',0) for x in history),'peak_worker_memory_mb':run['peak_worker_memory_mb'],'max_live_library_threads':run.get('max_live_worker_threads'),'configured_cpu_threads':8,'first_checkpoint_latency':run['first_checkpoint_latency'],'accepted':run['accepted_experiments'],'full_refit_published':run['full_refit_published'],'qwen_calls':run['qwen_calls']})
(ROOT/'reports/resource-reliability.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
