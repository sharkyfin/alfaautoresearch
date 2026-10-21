"""One guarded anytime research loop, with independent evaluation and checkpoints."""
from __future__ import annotations
import hashlib,json,os,signal,time,traceback,tempfile
from pathlib import Path
import numpy as np
from .utils import code_digest
from .contract import read_inputs
from .task import load,leakage_signals
from .submit import checkpoint,dummy_prediction
from .validation import design,confirmation_splits,Split
from .budget import Budget
from .experiments import Executor,register
from .recipes import validate,RecipeError,recipe_hash
from .research import controlled_hypothesis,ablation_options,accepts_ablation
from .search import SearchController,mock_hypothesis,retain_finalists,estimate_runtime
from .agent import QwenClient,researcher_context
from .judge import evaluate,meaningful_gain
from .ensemble import choose_blend,tune_multipliers,adjust_probabilities,robust_utility


def emit(event,**values):print(json.dumps({'event':event,**values},ensure_ascii=False,allow_nan=False),flush=True)

def run(task_path,started=None,budget_seconds=900,mode='auto',seed=42,threads=8,max_experiments=30,memory_mb=24576,run_dir=None,strict=True,research_client=None):
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
    control_plan=confirmation_splits(data,split,seed)
    data.spec.suspicious_leakage_features=leakage_signals(data,split.train)
    qwen=research_client if research_client is not None else QwenClient('live' if mode in {'auto','live-agent'} else 'disabled')
    research_origin=getattr(qwen,'origin','qwen');research_provider=getattr(qwen,'provider','organizer-qwen')
    actual_mode='live-agent' if qwen.configured and mode in {'auto','live-agent'} else 'mock-agent' if mode=='mock-agent' else 'deterministic'
    scheduler=SearchController(data.spec,budget,actual_mode,seed)
    executor=Executor(data,directory/'experiments',threads,memory_mb,lambda:stop['requested'],strict)
    initial_utility=None;no_gain=0;stage='SAFE_BASELINE';pending_ablation=None
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
        emit('CONFIRMATION_PLAN',windows=[s.json() for s in control_plan])
        # Search stops early enough to confirm and refit measured finalists.
        while len(history)<max_experiments and budget.search_remaining>2 and not stop['requested']:
            if time.monotonic()-start>budget.total*.67 and champion:break
            raw=None;feature_trial=None;feature_parent=None;ablation=None
            if pending_ablation and pending_ablation['checks']<3:
                for proposal,detail in ablation_options(champion['recipe'],pending_ablation['parent']['recipe'],pending_ablation['attempted']):
                    pending_ablation['attempted'].add(detail['label'])
                    if recipe_hash(proposal) in scheduler.seen:continue
                    raw=proposal;ablation=detail;break
            if ablation is not None:
                stage='FEATURE_ABLATION';is_research=False;origin='feature-ablation'
                feature_trial={'kind':'feature_ablation','group_experiment':pending_ablation['group_experiment'],
                               'parent_experiment':pending_ablation['parent']['experiment_id'],
                               'held_constant':['model_family','params','target_transform','validation','seed'],**ablation}
            else:
                pending_ablation=None
                raw=scheduler.next(champion);stage=scheduler.stage
                is_research=stage=='RESEARCH';origin='deterministic'
            if raw is None:break
            remaining=budget.search_remaining
            fallback=raw
            if is_research and champion:
                if actual_mode=='live-agent' and qwen.available:
                    context=researcher_context(data,history,champion,remaining)
                    context['recipe_rejections']=[e for e in qwen.events if e['status']=='recipe_rejected'][-3:]
                    proposed=qwen.propose(context,timeout=max(1,min(40,remaining*.25)))
                    if proposed is not None:raw=proposed;origin=research_origin
                elif actual_mode=='mock-agent' and scheduler.research_index<=3:
                    raw=mock_hypothesis(data.spec,champion,scheduler.research_index);origin='mock'
            attempts=[(raw,origin)]
            if origin==research_origin:attempts.append((fallback,'deterministic'))
            for raw,origin in attempts:
                try:
                    if not isinstance(raw,dict):raise RecipeError('Recipe must be an object')
                    requested=raw.get('max_runtime_sec',300)
                    if isinstance(requested,bool) or not isinstance(requested,(int,float)) or not 1<=requested<=1800:
                        raise RecipeError('Invalid time budget')
                    timeout=min(requested,max(2,budget.total*.15),budget.search_remaining)
                    raw={**raw,'max_runtime_sec':timeout,'validation_plan_id':split.id}
                    if is_research and champion:
                        feature_trial=None;feature_parent=None
                        r,feature_trial=controlled_hypothesis(raw,champion,data.spec,budget.search_remaining)
                        if feature_trial:feature_parent=champion
                    else:r=validate(raw,data.spec,remaining=budget.search_remaining)
                    r=validate(r,data.spec,remaining=budget.search_remaining,seen=scheduler.seen)
                    if scheduler.failures.get(r['model_family'],0)>=2:raise RecipeError('Model family repeatedly failed')
                    break
                except RecipeError as error:
                    register(directory/'rejections.jsonl',{'reason':str(error),'origin':origin})
                    if origin==research_origin:
                        qwen.events.append({'status':'recipe_rejected','reason':str(error)})
                        if sum(e['status']=='recipe_rejected' for e in qwen.events)>=3:qwen.open=True
            else:continue
            scheduler.seen.add(recipe_hash(r))
            # Reserve enough for a measured full-data refit before trying longer fits.
            refit_cost=champion['runtime_sec']*1.5 if champion else 0
            estimate=estimate_runtime(r,history,timeout,budget.total)
            final_work=len(control_plan)*sum(c['runtime_sec'] for c in finalists[:3])+1.5*sum(c['runtime_sec'] for c in finalists[:2])
            if champion and not budget.can_run(estimate,finalization_estimate=final_work):break
            timeout=min(timeout,max(1,budget.remaining-max(budget.reserve,refit_cost)))
            if ablation is not None:pending_ablation['checks']+=1
            result,predictions=executor.run(r,split,timeout,seed)
            result['origin']=origin;result['accepted']=False;result['estimated_runtime_sec']=estimate
            if feature_trial:result['feature_trial']=feature_trial
            if result['status']=='ok' and predictions is not None:
                try:
                    result.update(evaluate(data,split,predictions,result))
                    utility=result['standardized_utility']
                    reference=champion['standardized_utility'] if champion else initial_utility
                    accepted=reference is None or meaningful_gain(utility,reference)
                    if ablation is not None:
                        accepted=accepts_ablation(utility,reference,pending_ablation['parent']['standardized_utility'])
                    result['accepted']=accepted;result['accepted_reason']='objective gain on fixed holdout' if accepted else 'no meaningful objective gain'
                    if ablation is not None:result['accepted_reason']='feature change reverted without meaningful validation loss' if accepted else 'feature change retained after ablation'
                    if feature_trial and feature_trial['kind']=='feature_group':
                        feature_trial.update(group_experiment=result['experiment_id'],utility_gain_vs_parent=utility-feature_trial['parent_utility'])
                    candidate={**result,'predictions':predictions,'split':split}
                    family=r['model_family']
                    if accepted and ablation is not None:
                        finalists=[c for c in finalists if c['experiment_id']!=champion['experiment_id']]
                        if family_best.get(family,{}).get('experiment_id')==champion['experiment_id']:family_best[family]=candidate
                    if family not in family_best or utility>family_best[family]['standardized_utility']:family_best[family]=candidate
                    finalists=retain_finalists(finalists+[candidate]+list(family_best.values()))
                    if accepted:
                        champion=candidate;publish(champion);no_gain=0
                        if feature_parent is not None and len(feature_trial['changes'])>1:
                            pending_ablation={'parent':feature_parent,'group_experiment':result['experiment_id'],'attempted':set(),'checks':0}
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
            stage='CONFIRMATION'
            ordered=[champion]+[c for c in finalists if c['experiment_id']!=champion['experiment_id']]
            ordered=ordered[:3]
            for second in control_plan:
                if stop['requested']:break
                for candidate in ordered:
                    if stop['requested']:break
                    estimate=max(2,candidate['runtime_sec'])
                    if not budget.can_run(estimate,champion['runtime_sec']*1.3):break
                    r=dict(candidate['recipe']);r['validation_plan_id']=second.id
                    timeout=min(estimate*2+5,budget.search_remaining,max(2,budget.remaining*.3))
                    result,predictions=executor.run(r,second,timeout,second.seed)
                    result['accepted']=False;result['phase']='confirmation'
                    if result['status']=='ok' and predictions is not None:
                        try:
                            result.update(evaluate(data,second,predictions,result))
                            confirmed.setdefault(candidate['experiment_id'],[]).append({**result,'predictions':predictions,'split':second})
                        except ValueError as error:result.update(status='judge_error',message=str(error))
                    record(result)
            stable=[c for c in ordered if c['experiment_id'] in confirmed]
            common=set.intersection(*[{r['split'].id for r in confirmed[c['experiment_id']]} for c in stable]) if stable else set()
            common_evidence={c['experiment_id']:[r for r in confirmed[c['experiment_id']] if r['split'].id in common] for c in stable}
            if common:
                for candidate in stable:
                    evidence=common_evidence[candidate['experiment_id']]
                    candidate['robust_utility']=robust_utility([candidate['standardized_utility'],*[r['standardized_utility'] for r in evidence]])
                    candidate['confirmation_metrics']=[{'split_id':r['split'].id,'metric_value':r['metric_value']} for r in evidence]
                    candidate['confirmation_metric']=float(np.mean([r['metric_value'] for r in evidence]))
                if len(stable)>=2 and champion['experiment_id'] in common_evidence:
                    champion=max(stable,key=lambda c:c['robust_utility']);publish(champion)
                    stage='ENSEMBLE';blend=choose_blend(data,stable,common_evidence)
                    if blend:checkpoint(data,blend['predictions'][1]);emit('ensemble_accepted',weights=blend['weights'],score=blend['metric_value'],confirmation=blend['confirmation_metric'])
                if not blend:
                    multipliers=tune_multipliers(data,champion,common_evidence.get(champion['experiment_id']))
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
        summary['confirmation_plan']=[s.json() for s in control_plan]
        summary['confirmation_metrics']=champion.get('confirmation_metrics',[]) if champion else []
        summary['feature_group_trials']=sum(r.get('feature_trial',{}).get('kind')=='feature_group' for r in history)
        summary['feature_ablation_checks']=sum(r.get('feature_trial',{}).get('kind')=='feature_ablation' for r in history)
        summary['accepted_feature_ablations']=sum(r.get('accepted',False) and r.get('feature_trial',{}).get('kind')=='feature_ablation' for r in history)
        summary.update(researcher_provider=research_provider,researcher_model=qwen.model if qwen.configured else None,researcher_calls=qwen.requests,researcher_proposals=qwen.calls,researcher_tokens=qwen.tokens,researcher_events=qwen.events)
        if research_provider!='organizer-qwen':
            summary.update(qwen_calls=0,qwen_proposals=0,qwen_tokens=0,qwen_events=[])
        final_recipe={'output_sha256':summary['output_sha256'],'full_refit':refit_published,'seed':seed,'labels':data.spec.labels,'components':[{'recipe':c['recipe'],'iterations':c.get('iterations'),'experiment_id':c['experiment_id']} for c in (blend['components'] if blend else [champion] if champion else [])],'blend_weights':blend['weights'] if blend else [1.],'probability_multipliers':multipliers['weights'] if multipliers else None}
        (directory/'final_recipe.json').write_text(json.dumps(final_recipe,ensure_ascii=False,indent=2,allow_nan=False))
        (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        emit('DONE',summary=str(directory/'summary.json'),elapsed=round(summary['elapsed_seconds'],3),champion_metric=summary['champion_metric'])
