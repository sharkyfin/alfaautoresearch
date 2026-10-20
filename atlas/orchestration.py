"""One guarded anytime research loop, with independent evaluation and checkpoints."""
from __future__ import annotations
import hashlib,json,os,signal,time,traceback,tempfile
from pathlib import Path
import numpy as np
from .utils import code_digest
from .contract import read_inputs
from .task import load,leakage_signals
from .submit import checkpoint,dummy_prediction
from .validation import design,Split
from .budget import Budget
from .experiments import Executor,register
from .recipes import validate,RecipeError,recipe_hash
from .search import SearchController,mock_hypothesis,retain_finalists,estimate_runtime
from .agent import QwenClient,researcher_context
from .judge import evaluate,meaningful_gain
from .ensemble import choose_blend,tune_multipliers,adjust_probabilities


def emit(event,**values):print(json.dumps({'event':event,**values},ensure_ascii=False,allow_nan=False),flush=True)

def run(task_path,started=None,budget_seconds=10500,mode='auto',seed=42,threads=8,max_experiments=30,memory_mb=24576,run_dir=None,strict=True,research_client=None):
    # Install before the first atomic CSV becomes visible. A platform SIGTERM
    # immediately after that checkpoint must take the graceful path as well.
    stop={'requested':False}
    def handle_signal(signum,frame):stop['requested']=True
    old_handlers={s:signal.signal(s,handle_signal) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        return _run(task_path,started,budget_seconds,mode,seed,threads,max_experiments,memory_mb,run_dir,strict,research_client,stop)
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def _run(task_path,started,budget_seconds,mode,seed,threads,max_experiments,memory_mb,run_dir,strict,research_client,stop):
    source_digest=code_digest()
    start=started or time.monotonic();budget=Budget(float(budget_seconds),start)
    inputs=read_inputs(task_path);data=load(inputs,strict=strict)
    # Nothing expensive, stochastic or network-dependent precedes the first output.
    checkpoint(data,dummy_prediction(data.y,data.spec,len(data.test)))
    first_latency=time.monotonic()-start
    emit('SAFE_BASELINE',output=str(inputs.output),elapsed=first_latency,warnings=data.spec.warnings)
    if run_dir:
        directory=Path(run_dir).resolve()
        if directory.exists() and any(directory.iterdir()):raise ValueError('run_dir must be empty; choose a new directory for each run')
        directory.mkdir(parents=True,exist_ok=True)
    else:
        parent=inputs.task_path.parent/'.atlas';parent.mkdir(exist_ok=True)
        directory=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d_%H%M%S')+'_',dir=parent))
    (directory/'task_spec.json').write_text(json.dumps(data.spec.json(),ensure_ascii=False,indent=2,allow_nan=False))
    history=[];finalists=[];family_best={};champion=None;confirmed={};blend=None;multipliers=None;refit_published=False
    split=design(data,seed);data.spec.validation_plan=split.json()
    data.spec.suspicious_leakage_features=leakage_signals(data,split.train)
    qwen=research_client if research_client is not None else QwenClient('live' if mode in {'auto','live-agent'} else 'disabled')
    research_origin=getattr(qwen,'origin','qwen');research_provider=getattr(qwen,'provider','organizer-qwen')
    actual_mode='live-agent' if qwen.configured and mode in {'auto','live-agent'} else 'mock-agent' if mode=='mock-agent' else 'deterministic'
    scheduler=SearchController(data.spec,budget,actual_mode,seed)
    executor=Executor(data,directory/'experiments',threads,memory_mb,lambda:stop['requested'],strict)
    initial_utility=None;no_gain=0;stage='SAFE_BASELINE'
    def record(result):
        history.append(result);register(directory/'experiments.jsonl',result)
        emit('experiment',id=result['experiment_id'],family=result['recipe']['model_family'],status=result['status'],score=result.get('metric_value'),accepted=result.get('accepted',False),seconds=round(result['runtime_sec'],3))
    def publish(candidate):
        checkpoint(data,candidate['predictions'][1]);digest=hashlib.sha256(inputs.output.read_bytes()).hexdigest()
        meta={k:v for k,v in candidate.items() if k not in {'predictions','split'}}
        meta['output_sha256']=digest
        (directory/'champion.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False))
    try:
        if not len(split.valid) or len(np.unique(data.y))<2 or not data.spec.features:
            emit('degenerate_task',validation=split.json());return 0
        base=dummy_prediction(data.y[split.train],data.spec,len(split.valid))
        try:initial_utility=data.spec.metric.utility(data.spec.metric.score(data.y[split.valid],base))
        except ValueError as error:emit('metric_unavailable',reason=str(error));initial_utility=None
        (directory/'task_spec.json').write_text(json.dumps(data.spec.json(),ensure_ascii=False,indent=2,allow_nan=False))
        emit('VALIDATION',plan=split.json(),metric=data.spec.metric.name,mode=actual_mode)
        # Search stops early enough to confirm and refit measured finalists.
        while len(history)<max_experiments and budget.search_remaining>2 and not stop['requested']:
            if time.monotonic()-start>budget.total*.67 and champion:break
            raw=scheduler.next(champion);stage=scheduler.stage
            if raw is None:break
            is_research=stage=='RESEARCH'
            origin='deterministic'
            remaining=budget.search_remaining
            if is_research and champion:
                if actual_mode=='live-agent' and qwen.available:
                    context=researcher_context(data,history,champion,remaining)
                    context['recipe_rejections']=[e for e in qwen.events if e['status']=='recipe_rejected'][-3:]
                    proposed=qwen.propose(context,timeout=max(1,min(20,remaining*.1)))
                    if proposed is not None:raw=proposed;origin=research_origin
                elif actual_mode=='mock-agent' and scheduler.research_index<=3:
                    raw=mock_hypothesis(data.spec,champion,scheduler.research_index);origin='mock'
            # Already-failed libraries do not consume the rest of the budget.
            if scheduler.failures.get(raw.get('model_family'),0)>=2:continue
            timeout=min(float(raw.get('max_runtime_sec',300)),max(2,budget.total*.15),budget.search_remaining)
            raw['max_runtime_sec']=timeout;raw['validation_plan_id']=split.id
            try:r=validate(raw,data.spec,remaining=budget.search_remaining,seen=scheduler.seen)
            except RecipeError as error:
                register(directory/'rejections.jsonl',{'reason':str(error),'origin':origin})
                if origin==research_origin:
                    qwen.events.append({'status':'recipe_rejected','reason':str(error)})
                    if sum(e['status']=='recipe_rejected' for e in qwen.events)>=3:qwen.open=True
                continue
            scheduler.seen.add(recipe_hash(r))
            # Reserve enough for a measured full-data refit before trying longer fits.
            refit_cost=champion['runtime_sec']*1.5 if champion else 0
            estimate=estimate_runtime(r,history,timeout,budget.total)
            final_work=sum(c['runtime_sec'] for c in finalists[:3])+1.5*sum(c['runtime_sec'] for c in finalists[:2])
            if champion and not budget.can_run(estimate,finalization_estimate=final_work):break
            timeout=min(timeout,max(1,budget.remaining-max(budget.reserve,refit_cost)))
            result,predictions=executor.run(r,split,timeout,seed)
            result['origin']=origin;result['accepted']=False;result['estimated_runtime_sec']=estimate
            if result['status']=='ok' and predictions is not None:
                try:
                    result.update(evaluate(data,split,predictions,result))
                    utility=result['standardized_utility']
                    reference=champion['standardized_utility'] if champion else initial_utility
                    accepted=reference is None or meaningful_gain(utility,reference)
                    result['accepted']=accepted;result['accepted_reason']='objective gain on fixed holdout' if accepted else 'no meaningful objective gain'
                    candidate={**result,'predictions':predictions,'split':split}
                    family=r['model_family']
                    if family not in family_best or utility>family_best[family]['standardized_utility']:family_best[family]=candidate
                    finalists=retain_finalists(finalists+[candidate]+list(family_best.values()))
                    if accepted:champion=candidate;publish(champion);no_gain=0
                    else:no_gain+=1
                except (ValueError,ArithmeticError) as error:
                    result.update(status='judge_error',exception_class=type(error).__name__,message=str(error));scheduler.failures[r['model_family']]=scheduler.failures.get(r['model_family'],0)+1
            else:
                scheduler.failures[r['model_family']]=scheduler.failures.get(r['model_family'],0)+1
            record(result)
            keep_ids={c['experiment_id'] for c in finalists+list(family_best.values())}
            if champion: keep_ids.add(champion['experiment_id'])
            for artifact in executor.root.glob('exp_*/predictions.npz'):
                if artifact.parent.name not in keep_ids:artifact.unlink()
            if no_gain>=10:break
        if champion and not stop['requested']:
            stage='CONFIRMATION';second=design(data,seed+1009,fraction=.25)
            if second.id!=split.id and len(second.valid):
                for candidate in finalists[:3]:
                    estimate=max(2,candidate['runtime_sec'])
                    if not budget.can_run(estimate,champion['runtime_sec']*1.3):break
                    r=dict(candidate['recipe']);r['validation_plan_id']=second.id
                    timeout=min(estimate*2+5,budget.search_remaining,max(2,budget.remaining*.3))
                    result,predictions=executor.run(r,second,timeout,seed+1009)
                    result['accepted']=False;result['phase']='confirmation'
                    if result['status']=='ok' and predictions is not None:
                        try:
                            result.update(evaluate(data,second,predictions,result));confirmed[candidate['experiment_id']]={**result,'predictions':predictions,'split':second}
                            utilities=[candidate['standardized_utility'],result['standardized_utility']]
                            candidate['robust_utility']=float(np.mean(utilities)-.25*np.std(utilities))
                            candidate['confirmation_metric']=result['metric_value']
                        except ValueError as error:result.update(status='judge_error',message=str(error))
                    record(result)
                stable=[c for c in finalists if c['experiment_id'] in confirmed]
                if len(stable)>=2:
                    champion=max(stable,key=lambda c:c['robust_utility']);publish(champion)
                    stage='ENSEMBLE';blend=choose_blend(data,stable,confirmed)
                    if blend:checkpoint(data,blend['predictions'][1]);emit('ensemble_accepted',weights=blend['weights'],score=blend['metric_value'],confirmation=blend['confirmation_metric'])
                if not blend:
                    multipliers=tune_multipliers(data,champion,confirmed.get(champion['experiment_id']))
                    if multipliers:checkpoint(data,multipliers['predictions'][1]);emit('probability_adjustment_accepted',score=multipliers['metric_value'],confirmation=multipliers['confirmation_metric'])
        # Publish refit only after every component succeeds; previous good file survives.
        if champion and not stop['requested']:
            stage='FINALIZE';components=blend['components'] if blend else [champion];weights=blend['weights'] if blend else [1.]
            refitted=[];full_split=Split(np.arange(len(data.y)),np.empty(0,dtype=int),'full_refit','All known labels after selection',seed)
            for c in components:
                estimate=max(2,c['runtime_sec']*1.6)
                if budget.remaining<max(1.5,estimate*.45):break
                timeout=max(1,min(estimate*2+5,budget.remaining-min(5,budget.reserve*.25)))
                result,predictions=executor.run(c['recipe'],full_split,timeout,seed,refit=True,iterations_override=c.get('iterations'))
                result['accepted']=False;result['phase']='refit'
                record(result)
                if result['status']!='ok' or predictions is None:break
                refitted.append(predictions[1])
            if len(refitted)==len(components):
                final=sum(w*p for w,p in zip(weights,refitted))
                if multipliers:final=adjust_probabilities(final,multipliers['weights'])
                checkpoint(data,final);refit_published=True;emit('full_refit_published',components=len(components))
        return 0
    except Exception as error:
        # The valid atomic output exists. Keep a diagnostic and exit gracefully.
        emit('orchestrator_error',exception_class=type(error).__name__,message=str(error)[:500])
        (directory/'error.txt').write_text(traceback.format_exc())
        return 0 if inputs.output.exists() else 1
    finally:
        summary={'runtime_code_sha256':source_digest,'mode':actual_mode,'budget_seconds':budget.total,'elapsed_seconds':time.monotonic()-start,'first_checkpoint_latency':first_latency,'validation':split.json(),'metric':data.spec.metric.name,'dummy_utility':initial_utility,'experiments':len(history),'successful_experiments':sum(r['status']=='ok' for r in history),'accepted_experiments':sum(r.get('accepted',False) for r in history),'champion_metric':champion['metric_value'] if champion else None,'champion_recipe':champion['recipe'] if champion else None,'confirmation_metric':champion.get('confirmation_metric') if champion else None,'ensemble':{k:v for k,v in blend.items() if k not in {'components','predictions'}} if blend else None,'multipliers':{k:v for k,v in multipliers.items() if k!='predictions'} if multipliers else None,'qwen_calls':qwen.requests,'qwen_proposals':qwen.calls,'qwen_tokens':qwen.tokens,'qwen_events':qwen.events,'max_live_worker_threads':max((r.get('max_live_threads',0) for r in history),default=0),'peak_worker_memory_mb':max((r['peak_memory_mb'] for r in history),default=0),'cancelled':stop['requested'],'output':str(inputs.output),'output_sha256':hashlib.sha256(inputs.output.read_bytes()).hexdigest() if inputs.output.exists() else None}
        summary['full_refit_published']=refit_published
        summary.update(researcher_provider=research_provider,researcher_model=qwen.model if qwen.configured else None,researcher_calls=qwen.requests,researcher_proposals=qwen.calls,researcher_tokens=qwen.tokens,researcher_events=qwen.events)
        if research_provider!='organizer-qwen':
            summary.update(qwen_calls=0,qwen_proposals=0,qwen_tokens=0,qwen_events=[])
        final_recipe={'output_sha256':summary['output_sha256'],'full_refit':refit_published,'seed':seed,'labels':data.spec.labels,'components':[{'recipe':c['recipe'],'iterations':c.get('iterations'),'experiment_id':c['experiment_id']} for c in (blend['components'] if blend else [champion] if champion else [])],'blend_weights':blend['weights'] if blend else [1.],'probability_multipliers':multipliers['weights'] if multipliers else None}
        (directory/'final_recipe.json').write_text(json.dumps(final_recipe,ensure_ascii=False,indent=2,allow_nan=False))
        (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        emit('DONE',summary=str(directory/'summary.json'),elapsed=round(summary['elapsed_seconds'],3),champion_metric=summary['champion_metric'])
