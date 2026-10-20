"""Summarize measured evidence without inventing platform scores or live Qwen results."""
import hashlib,json,re,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.utils import code_digest
reports=ROOT/'reports'
text=(reports/'all-tests.log').read_text();match=re.search(r'(\d+) passed(?:, (\d+) warnings)? in ([\d.]+)s',text)
if 'FAILED ' in text:match=None
lines=['# Evaluation evidence','',f'Current runtime source SHA256: `{code_digest()}`.','',f'Automated suite: **{match.group(1)} passed**, {match.group(2) or 0} warnings in {match.group(3)}s.' if match else 'Automated suite has no verified final passing summary.','Command: `.venv/bin/python -m pytest tests -q --tb=short`.','Warnings in the current suite are upstream Matplotlib/pyparsing deprecations. Historical failure logs are retained; all-tests.log is the current result.','', '## Public task measurements','', 'These are internal train-label holdouts, not organizer baseline scores or leaderboard estimates. Hidden test labels are never read.','', '| Task / run | Seed | Split | Linear reference | Selected objective | Confirmation objective | Seconds | Peak worker MiB | Accepted |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
def selected(r):return (r.get('ensemble') or r.get('multipliers') or {}).get('metric_value',r['champion_metric'])
def control(r):return (r.get('ensemble') or r.get('multipliers') or {}).get('confirmation_metric',r.get('confirmation_metric'))
public=[]
for kind,name in [('regression','public-reg-360'),('classification','public-cls-360')]:
 p=ROOT/'runs'/name/'summary.json'
 if p.exists():r=json.loads(p.read_text());r.update(task=kind,seed=42,run=name);public.append(r)
for campaign in ['public-campaign','public-final']:
 p=reports/f'{campaign}.json'
 if p.exists():
  for r in json.loads(p.read_text()):r['run']=f"{campaign}/{r['task']}_{r['seed']}";public.append(r)
for r in public:
 history=[json.loads(s) for s in (ROOT/'runs'/r['run']/'experiments.jsonl').read_text().splitlines()]
 reference=next((x['metric_value'] for x in history if x['recipe']['model_family']=='linear' and 'metric_value' in x),None)
 def num(x):return 'n/a' if x is None else f'{x:.6g}'
 lines.append(f"| {r['task']} / {r['run']} | {r['seed']} | {r['validation']['kind']} | {num(reference)} | {num(selected(r))} | {num(control(r))} | {r['elapsed_seconds']:.1f} | {r['peak_worker_memory_mb']:.1f} | {r['accepted_experiments']} |")
lines+=['','The second split is independent for selection confirmation; its absolute score is not directly compared to the first split. The two 360s exploratory runs predate the frozen source campaign. Each run preserves generated code, actual predictions, recipe hashes, source hashes, fixed splits, diagnostics and final recipe.','', '## Synthetic private-like coverage','']
p=reports/'stress-final/results.json'
if not p.exists():p=reports/'stress/results.json'
if p.exists():
 r=json.loads(p.read_text());valid=[x for x in r if x['status']=='valid_output'];unsupported=[x for x in r if x['status']=='unsupported'];failed=[x for x in r if x['status'] not in {'unsupported','valid_output'}];scored=[x for x in valid if 'gain_utility' in x]
 lines += [f"35 scenarios × 2 seeds: {len(valid)} valid outputs, {len(unsupported)} explicitly expected unsupported tasks, {len(failed)} unexpected failures.",f"Among {len(scored)} scored nondegenerate tasks: {sum(x['gain_utility']>0 for x in scored)} gains, {sum(x['gain_utility']<0 for x in scored)} losses, {sum(x['gain_utility']==0 for x in scored)} ties versus a training-fold dummy. No raw RMSE/F1 averaging is used.",'','| Scenario | Seed | Validation | Metric | Dummy | Model | Utility gain | Seconds |','|---|---:|---|---|---:|---:|---:|---:|']
 for x in scored:lines.append(f"| {x['scenario']} | {x['seed']} | {x['split']['kind']} | {x['metric']} | {x['baseline']:.6g} | {x['score']:.6g} | {x['gain_utility']:.6g} | {x['seconds']:.3f} |")
 lines+=['','Ambiguous dictionary targets and unsupported multi-target samples are classified as unsupported; these cases are not counted as successful model runs. Constant-target and one-class tasks receive a valid constant prediction.']
lines+=['','## Live equal-budget comparisons','']
for name,provider in [('openrouter-paired-v2','openrouter'),*[(p.stem,'organizer') for p in sorted(reports.glob('organizer-paired-*.json'))]]:
 p=reports/f'{name}.json'
 if not p.exists():continue
 campaign=json.loads(p.read_text());manifest=campaign['manifest'];paired=[r for r in campaign['pairs'] if r['status']=='compared']
 lines += [f"### {provider}: {campaign['status']}",'',f"Frozen runtime `{manifest['runtime_code_sha256']}`; {manifest['budget_seconds']:g}s per run, {manifest['threads']} configured threads, max {manifest['max_experiments']} exploration fits. Same data, validation, seed and configured limits in each pair; variants run sequentially. The max-experiment cap can finish before the time limit.",'', '| Task | Seed | Deterministic | Live | Normalized gain | Deterministic seconds | Live seconds | Live fits / accepted |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for row in paired:
  lines.append(f"| {row['task']} | {row['seed']} | {row['deterministic']:.6g} | {row[provider]:.6g} | {row['relative_improvement']:.2%} | {row['deterministic_seconds']:.2f} | {row.get('live_seconds',row.get('openrouter_seconds',0)):.2f} | {row['live_fits']} / {row['live_accepted']} |")
 if paired:
  gains=[r['relative_improvement'] for r in paired]
  lines += ['',f"{len(gains)} pairs: {sum(x>0 for x in gains)} wins, {sum(x<0 for x in gains)} losses, {sum(x==0 for x in gains)} ties; median normalized improvement {statistics.median(gains):.2%}. No statistical significance claim."]
 if provider=='openrouter':lines+=['Reported cost for this campaign was zero. Three wins, three losses and two ties do not establish consistent quality benefit. Some live proposals displace useful deterministic exploration under the fixed attempt cap; this motivates reducing research frequency.']
 lines+=['']
lines+=['','## Historical mock infrastructure comparison','', 'The following older mock experiment cannot establish Qwen research value. Its source snapshot differs from the current live experiments.','']
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
 differences=[]
 for pair in pairs.values():
  if not {'deterministic','mock-agent'}<=set(pair):continue
  d,m=pair['deterministic'],pair['mock-agent'];direction=-1 if d['metric'] in {'rmse','mae','mse','rmsle','logloss'} else 1
  differences.append(direction*(selected(m)-selected(d))/max(abs(selected(d)),1e-12))
 if differences:
  quartile=statistics.quantiles(differences,n=4,method='inclusive')[0] if len(differences)>1 else differences[0]
  lines += ['',f"Across {len(differences)} prototype pairs: {sum(x>0 for x in differences)} wins, {sum(x<0 for x in differences)} losses, {sum(x==0 for x in differences)} ties; win rate {sum(x>0 for x in differences)/len(differences):.1%}. Median normalized improvement {statistics.median(differences):.3%}, lower quartile {quartile:.3%}. Normalization is signed objective difference divided by the absolute deterministic objective on each task; this is not the competition scoring formula."]
p=reports/'openrouter-live.json'
if p.exists():
 r=json.loads(p.read_text());s=r['summary'];hypotheses=r['live_hypotheses']
 lines+=['','## User-authorized OpenRouter development run','',f"Pinned model: `{r['model']}`; {s['researcher_calls']} live calls, {s['researcher_tokens']} total tokens, ${r['loop_cost_usd']:g} reported loop cost. Runtime source: `{r['runtime_code_sha256']}`. The separate JSON protocol smoke also reported zero cost.",f"On one synthetic regression task (seed 42, 500 train rows), both real hypotheses compiled and fitted successfully; {sum(x['accepted'] for x in hypotheses)} improved the initial champion. Initial champion RMSE {s['champion_metric']:.6g}; hypothesis RMSE values: "+', '.join(f"{x['metric_value']:.6g}" for x in hypotheses)+'.',f"Final selected ensemble RMSE {s['ensemble']['metric_value']:.6g}; output CSV validated, all full refits succeeded, elapsed {s['elapsed_seconds']:.2f}s. This is a working live backend test, not evidence of research quality gain or an equal-budget ablation.",'The first protocol probe rejected reasoning.enabled=false with HTTP400; low reasoning effort and 4096 output-token allowance resolved it. The model is pinned to :free and provider input/output maximum prices are zero. No payment method or credits were added. Key values and the development client are excluded from the competition ZIP.']
lines+=['','## Leakage and failure tests','']
p=reports/'leakage-ablation.json'
if p.exists():
 for x in json.loads(p.read_text()):lines.append(f"- {x['scenario']}: {x['validation']}, RMSE {x['score']:.6g}.")
lines+=['','Actual kill -9 during fit, SIGTERM orchestration, memory-limit termination, worker timeout, invalid output rejection and kill immediately before os.replace are covered by tests/test_execution.py and tests/test_core.py. The previous CSV remains valid.','', 'Scaled 3h scheduling simulation: nominal 10800s × 1/180 = 60s actual budget; see deadline-simulation.json. This is **not a real 3-hour soak**.','', '## Release limits','', '- macOS development host: 12 logical CPUs, 16 GiB RAM; worker threads capped at 8.','- Linux container verification is recorded separately in linux-verification.json when available. A local 16GiB host cannot reproduce physical 32GiB conditions.','- The new archive has only root Python files and official requirements.txt. verify_zip.py reproduces the platform copy filter and exact task cwd; its current results are in zip-verification.json.','- No platform submission has been made. Daily team submission allowance cannot be inferred locally.','- Pretrained models/external training datasets are absent from runtime and ZIP.','', 'The next quality experiment should preserve the deterministic exploration queue while reducing LLM frequency; verify its effect on skewed regression and minority-class F1 under equal budgets. Local verification status and remaining gates are in final-release-audit.json; no platform score is available.']
(reports/'EVALUATION.md').write_text('\n'.join(lines)+'\n')
print(reports/'EVALUATION.md')
