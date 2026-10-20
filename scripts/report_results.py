"""Summarize measured evidence without inventing platform scores or live Qwen results."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.utils import code_digest
reports=ROOT/'reports'
text=(reports/'all-tests.log').read_text();match=re.search(r'(\d+) passed(?:, (\d+) warnings)? in ([\d.]+)s',text)
lines=['# Evaluation evidence','',f'Current runtime source SHA256: `{code_digest()}`.','',f'Automated suite: **{match.group(1)} passed**, {match.group(2) or 0} warnings in {match.group(3)}s.' if match else 'Automated suite has no verified final passing summary.','Command: `.venv/bin/python -m pytest tests -q --tb=short`.','Warnings in the current suite are upstream Matplotlib/pyparsing deprecations. Historical failure logs are retained; all-tests.log is the current result.','', '## Public task measurements','', 'These are internal train-label holdouts, not organizer baseline scores or leaderboard estimates. Hidden test labels are never read.','', '| Task / run | Seed | Split | Linear reference | Selected objective | Confirmation objective | Seconds | Peak worker MiB | Accepted |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
def selected(r):return (r.get('ensemble') or r.get('multipliers') or {}).get('metric_value',r['champion_metric'])
def control(r):return (r.get('ensemble') or r.get('multipliers') or {}).get('confirmation_metric',r.get('confirmation_metric'))
public=[]
for kind,name in [('regression','public-reg-360'),('classification','public-cls-360')]:
 p=ROOT/'runs'/name/'summary.json'
 if p.exists():r=json.loads(p.read_text());r.update(task=kind,seed=42,run=name);public.append(r)
p=reports/'public-campaign.json'
if p.exists():
 for r in json.loads(p.read_text()):r['run']=f"public-campaign/{r['task']}_{r['seed']}";public.append(r)
for r in public:
 history=[json.loads(s) for s in (ROOT/'runs'/r['run']/'experiments.jsonl').read_text().splitlines()]
 reference=next((x['metric_value'] for x in history if x['recipe']['model_family']=='linear' and 'metric_value' in x),None)
 def num(x):return 'n/a' if x is None else f'{x:.6g}'
 lines.append(f"| {r['task']} / {r['run']} | {r['seed']} | {r['validation']['kind']} | {num(reference)} | {num(selected(r))} | {num(control(r))} | {r['elapsed_seconds']:.1f} | {r['peak_worker_memory_mb']:.1f} | {r['accepted_experiments']} |")
lines+=['','The second split is independent for selection confirmation; its absolute score is not directly compared to the first split. The two 360s exploratory runs predate the frozen source campaign. Each run preserves generated code, actual predictions, recipe hashes, source hashes, fixed splits, diagnostics and final recipe.','', '## Synthetic private-like coverage','']
p=reports/'stress/results.json'
if p.exists():
 r=json.loads(p.read_text());valid=[x for x in r if x['status']=='valid_output'];unsupported=[x for x in r if x['status']=='unsupported'];failed=[x for x in r if x['status'] not in {'unsupported','valid_output'}];scored=[x for x in valid if 'gain_utility' in x]
 lines += [f"35 scenarios × 2 seeds: {len(valid)} valid outputs, {len(unsupported)} explicitly expected unsupported tasks, {len(failed)} unexpected failures.",f"Among {len(scored)} scored nondegenerate tasks: {sum(x['gain_utility']>0 for x in scored)} gains, {sum(x['gain_utility']<0 for x in scored)} losses, {sum(x['gain_utility']==0 for x in scored)} ties versus a training-fold dummy. No raw RMSE/F1 averaging is used.",'','| Scenario | Seed | Validation | Metric | Dummy | Model | Utility gain | Seconds |','|---|---:|---|---|---:|---:|---:|---:|']
 for x in scored:lines.append(f"| {x['scenario']} | {x['seed']} | {x['split']['kind']} | {x['metric']} | {x['baseline']:.6g} | {x['score']:.6g} | {x['gain_utility']:.6g} | {x['seconds']:.3f} |")
 lines+=['','Ambiguous dictionary targets and unsupported multi-target samples are classified as unsupported; these cases are not counted as successful model runs. Constant-target and one-class tasks receive a valid constant prediction.']
lines+=['','## Equal-budget ablation','', '**Live Qwen ablation is BLOCKED:** user confirmed organizer credentials/endpoint/model settings are not available. Zero live calls were made. The following is a mock-infrastructure comparison and cannot establish Qwen research value.','']
p=reports/'paired-ablation.json'
if p.exists():
 r=json.loads(p.read_text());pairs={}
 for x in r:pairs.setdefault((x['task'],x['seed']),{})[x['mode']]=x
 lines+=['| Task | Seed | Deterministic | Mock agent | Utility difference | Deterministic seconds | Mock seconds |','|---|---:|---:|---:|---:|---:|---:|']
 for (task,seed),pair in pairs.items():
  if not {'deterministic','mock-agent'}<=set(pair):continue
  d,m=pair['deterministic'],pair['mock-agent'];direction=-1 if d['metric'] in {'rmse','mae','mse','rmsle','logloss'} else 1
  lines.append(f"| {task} | {seed} | {selected(d):.6g} | {selected(m):.6g} | {direction*(selected(m)-selected(d)):.6g} | {d['elapsed_seconds']:.2f} | {m['elapsed_seconds']:.2f} |")
 lines+=['','Each finalized pair uses the prototype runtime SHA256 recorded in paired-ablation.json. Later safety fixes and portfolio additions are tested separately. Each pair uses the same data, seed, budget and configured CPU threads. Source is frozen in runs/paired/source_snapshot. Earlier mixed-revision runs are separately labeled exploratory and excluded from this table. Sample size is insufficient for statistical significance claims.']
lines+=['','## Leakage and failure tests','']
p=reports/'leakage-ablation.json'
if p.exists():
 for x in json.loads(p.read_text()):lines.append(f"- {x['scenario']}: {x['validation']}, RMSE {x['score']:.6g}.")
lines+=['','Actual kill -9 during fit, SIGTERM orchestration, memory-limit termination, worker timeout, invalid output rejection and kill immediately before os.replace are covered by tests/test_execution.py and tests/test_core.py. The previous CSV remains valid.','', 'Scaled 3h scheduling simulation: nominal 10800s × 1/180 = 60s actual budget; see deadline-simulation.json. This is **not a real 3-hour soak**.','', '## Release limits','', '- macOS development host: 12 logical CPUs, 16 GiB RAM; worker threads capped at 8.','- Docker is unavailable locally. Linux RLIMIT_AS path and exact 8-CPU/32-GiB Docker environment still require verification.','- Archive was unpacked into a clean temporary directory and the official root CLI was exercised for regression and multiclass, with SIGTERM after a fitted checkpoint.','- No platform submission has been made. Daily team submission allowance cannot be inferred locally.','- Pretrained models/external training datasets are absent from runtime and ZIP.','', 'The next most valuable experiment is a live organizer-Qwen paired ablation on the frozen suite at equal time budgets; first verify its endpoint and JSON protocol. Until settings are available, the archive remains explicitly provisional.']
(reports/'EVALUATION.md').write_text('\n'.join(lines)+'\n')
print(reports/'EVALUATION.md')
