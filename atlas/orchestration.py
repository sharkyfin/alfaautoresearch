"""One guarded anytime research loop, with independent evaluation and checkpoints."""
from __future__ import annotations
import hashlib,json,os,signal,time,traceback,tempfile
from pathlib import Path
import numpy as np
from .utils import code_digest
from . import challenger as automl
from .metrics import UnsupportedTask
from .contract import read_inputs
from .task import load,leakage_signals
from .submit import checkpoint,dummy_prediction
from .validation import research_splits as selection_splits,Split,split_index_digest,confirmation_batches
from .budget import Budget
from .experiments import Executor,register
from .recipes import validate,RecipeError,recipe_hash
from .research import controlled_hypothesis,ablation_options,accepts_ablation
from .search import SearchController,mock_hypothesis,retain_finalists,estimate_runtime,resolve_strategy
from .agent import QwenClient,researcher_context
from .judge import evaluate,meaningful_gain
from .ensemble import choose_blend,tune_multipliers,adjust_probabilities,robust_utility,store_oof


def emit(event,**values):print(json.dumps({'event':event,**values},ensure_ascii=False,allow_nan=False),flush=True)

def run(task_path,started=None,budget_seconds=10500,mode='auto',seed=42,threads=8,max_experiments=240,memory_mb=24576,run_dir=None,strict=True,research_client=None,ensemble_strategy='greedy',finalization='refit'):
    if ensemble_strategy not in {'none','weighted','greedy'} or finalization not in {'refit','fold_average'}:
        raise ValueError('Invalid ensemble/finalization policy')
    if run_dir and Path(run_dir).exists() and any(Path(run_dir).iterdir()):
        raise ValueError('run_dir must be empty; choose a new directory for each run')
    # Install before the first atomic CSV becomes visible. A platform SIGTERM
    # immediately after that checkpoint must take the graceful path as well.
    stop={'requested':False}
    def handle_signal(signum,frame):stop['requested']=True
    old_handlers={s:signal.signal(s,handle_signal) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        return _run(task_path,started,budget_seconds,mode,seed,threads,max_experiments,memory_mb,run_dir,strict,research_client,stop,ensemble_strategy,finalization)
    except Exception as error:
        # Failures before the main loop also need provenance. Do not overwrite
        # a supplied, nonempty run_dir or claim an older CSV is a valid fallback.
        parent=Path(task_path).resolve().parent/'.atlas';parent.mkdir(parents=True,exist_ok=True)
        directory=Path(tempfile.mkdtemp(prefix='setup_failure_',dir=parent))
        summary={'runtime_code_sha256':code_digest(),'status':'unsupported_task' if isinstance(error,UnsupportedTask) else 'fatal_setup',
            'terminal_phase':'setup','checkpoint_kind':'unverified_existing_output' if (parent.parent/'submission.csv').exists() else 'missing',
            'output_exists':(parent.parent/'submission.csv').exists(),'trained_output':False,'fit_success':0,
            'budget_seconds':float(budget_seconds),'fatal_exception':{'exception_class':type(error).__name__,'message':str(error)[:500]}}
        (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        emit('orchestrator_setup_error',status=summary['status'],summary=str(directory/'summary.json'))
        return 2 if isinstance(error,UnsupportedTask) else 1
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def _run(task_path,started,budget_seconds,mode,seed,threads,max_experiments,memory_mb,run_dir,strict,research_client,stop,ensemble_strategy,finalization):
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
    from .ideas import idea_registry
    (directory/'idea_registry.json').write_text(json.dumps(idea_registry(data.spec),indent=2))
    history=[];finalists=[];family_best={};champion=None;confirmed={};blend=None;multipliers=None;refit_published=False;fold_bagged_published=False;cv_selection={}
    checkpoint_kind='valid_dummy';fatal_error=None
    challenger_state={'eligible':False,'reason':'no_trained_champion','reserve_seconds':0.};challenger_candidate=None;challenger_planned=False
    split,control_plan=selection_splits(data,seed);data.spec.validation_plan=split.json()
    control_jobs=confirmation_batches(control_plan,seed)
    control_fits=len(control_jobs)
    if control_plan:
        pool=np.union1d(split.train,split.valid)
        data.spec.validation_plan.update(confirmation_isolation='sealed',search_pool_sha256=split_index_digest(pool),
            search_pool_rows=len(pool),confirmation_train_sha256=split_index_digest(control_plan[0].train))
        target=data.y[pool]
        if data.spec.classification:
            labels,counts=np.unique(target,return_counts=True)
            data.spec.profile['target']={'counts':{str(data.spec.labels[int(label)]):int(count) for label,count in zip(labels,counts)}}
        else:data.spec.profile['target']={'quantiles':{str(q):float(v) for q,v in zip([0,.01,.25,.5,.75,.99,1],np.quantile(target,[0,.01,.25,.5,.75,.99,1]))}}
        data.spec.profile['target_scope']='Search pool only; sealed control targets excluded'

    data.spec.suspicious_leakage_features=leakage_signals(data,split.train)
    qwen=research_client if research_client is not None else QwenClient('live' if mode in {'auto','live-agent'} else 'disabled')
    research_origin=getattr(qwen,'origin','qwen');research_provider=getattr(qwen,'provider','organizer-qwen')
    actual_mode='live-agent' if qwen.configured and mode in {'auto','live-agent'} else 'mock-agent' if mode=='mock-agent' else 'deterministic'
    scheduler=SearchController(data.spec,budget,actual_mode,seed)
    executor=Executor(data,directory/'experiments',threads,memory_mb,lambda:stop['requested'],strict)
    initial_utility=None;no_gain=0;stage='SAFE_BASELINE';pending_ablation=None
    def record(result):
        if not result.get('phase'):scheduler.observe(result)
        history.append(result);register(directory/'experiments.jsonl',result)
        emit('experiment',id=result['experiment_id'],family=result['recipe']['model_family'],status=result['status'],score=result.get('metric_value'),accepted=result.get('accepted',False),seconds=round(result['runtime_sec'],3))
    def run_candidate(candidate,window,timeout,candidate_seed,**kwargs):
        if candidate['recipe']['model_family']==automl.FAMILY:
            kwargs.update(challenger_policy=candidate['challenger_policy'],source_artifact=candidate['artifact_path'])
        return executor.run(candidate['recipe'],window,timeout,candidate_seed,**kwargs)
    def publish(candidate):
        nonlocal checkpoint_kind
        checkpoint(data,candidate['predictions'][1]);digest=hashlib.sha256(inputs.output.read_bytes()).hexdigest()
        checkpoint_kind='confirmed_champion' if candidate.get('confirmation_metrics') else 'trained_checkpoint'
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
            scheduler.research_enabled=actual_mode=='live-agent' and qwen.available
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
                is_research=stage=='RESEARCH';origin='optuna-tpe' if stage=='HYPERPARAMETER_SEARCH' else 'deterministic'
            if raw is None:break
            remaining=budget.search_remaining
            fallback=raw
            if is_research and champion:
                if actual_mode=='live-agent' and qwen.available and scheduler.research_due(champion):
                    measured=[event['elapsed_seconds'] for event in qwen.events
                        if event.get('status')=='api_usage' and 'elapsed_seconds' in event]
                    api_estimate=max(3.,float(np.median(measured))) if measured else 8.
                    probe_cost=max(1.,champion.get('runtime_sec',1.)*1.3)
                    final_work=control_fits*sum(c['runtime_sec'] for c in finalists[:3])+1.5*sum(c['runtime_sec'] for c in finalists[:2])+challenger_state['reserve_seconds']
                    if budget.can_run(probe_cost+api_estimate,finalization_estimate=final_work):
                        context=researcher_context(data,history,champion,remaining)
                        context['research_evidence']['api_latency_median_seconds']=api_estimate
                        context['recipe_rejections']=[e for e in qwen.events if e['status']=='recipe_rejected'][-3:]
                        api_slack=budget.remaining-max(budget.reserve,final_work)-probe_cost*1.3
                        scheduler.note_research(champion)
                        proposed=qwen.propose(context,timeout=max(1,min(40,remaining*.25,api_slack)))
                        if proposed is not None:raw=proposed;origin=research_origin
                    else:qwen.events.append({'status':'skipped_finalization_reserve','remaining_seconds':round(budget.remaining,3),
                        'estimated_api_seconds':api_estimate,'estimated_fit_seconds':probe_cost,'estimated_finalization_seconds':final_work})
                elif actual_mode=='mock-agent' and scheduler.research_index<=3:
                    raw=mock_hypothesis(data.spec,champion,scheduler.research_index);origin='mock'
            attempts=[(raw,origin)]
            if origin==research_origin:attempts.append((fallback,'deterministic'))
            research_strategy=None
            for raw,origin in attempts:
                try:
                    raw,research_strategy=resolve_strategy(raw,scheduler,champion,history,fallback,budget.search_remaining) if champion and origin==research_origin else (raw,None)
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
            if research_strategy and research_strategy.get('allocation'):scheduler.set_focus(research_strategy['allocation'])
            scheduler.seen.add(recipe_hash(r))
            # Reserve enough for a measured full-data refit before trying longer fits.
            refit_cost=champion['runtime_sec']*1.5 if champion else 0
            estimate=estimate_runtime(r,history,timeout,budget.total)
            if champion and not challenger_planned:
                challenger_state=automl.plan(data,champion,budget,control_plan);challenger_planned=True
            final_work=control_fits*sum(c['runtime_sec'] for c in finalists[:3])+1.5*sum(c['runtime_sec'] for c in finalists[:2])+challenger_state['reserve_seconds']
            if champion and not budget.can_run(estimate,finalization_estimate=final_work):break
            timeout=min(timeout,max(1,budget.remaining-max(budget.reserve,refit_cost)))
            if ablation is not None:pending_ablation['checks']+=1
            result,predictions=executor.run(r,split,timeout,seed)
            result['reference_utility']=champion['standardized_utility'] if champion else initial_utility
            if research_strategy:result['research_strategy']=research_strategy
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
            if scheduler.should_stop(no_gain):break
        scheduler.hpo.abandon()
        if champion and not stop['requested']:
            stage='AUTOML_CHALLENGER'
            challenger_state=automl.plan(data,champion,budget,control_plan)
            if challenger_state['eligible']:
                timeout=challenger_state['worker_seconds']
                r=automl.recipe(timeout,split.id)
                result,predictions=executor.run(r,split,timeout,seed)
                result.update(phase='challenger',origin='bounded-autogluon',accepted=False)
                if result['status']=='ok' and predictions is not None:
                    try:
                        result.update(evaluate(data,split,predictions,result))
                        challenger_candidate={**result,'predictions':predictions,'split':split}
                        challenger_state.update(status='awaiting_independent_controls',metric_value=result['metric_value'])
                    except (ValueError,ArithmeticError) as error:
                        result.update(status='judge_error',message=str(error))
                if not challenger_candidate:challenger_state.update(status='failed',failure=result.get('message',result['status']))
                record(result)
            else:challenger_state['status']='skipped'
        if champion and not stop['requested']:
            stage='CONFIRMATION'
            ordered=[champion]+[c for c in finalists if c['experiment_id']!=champion['experiment_id']]
            ordered=ordered[:3]+([challenger_candidate] if challenger_candidate else [])
            for batch,windows in control_jobs:
                if stop['requested']:break
                for candidate in ordered:
                    if stop['requested']:break
                    estimate=max(2,candidate['runtime_sec'])*1.4
                    if not budget.can_run(estimate,champion['runtime_sec']*1.3):break
                    r=dict(candidate['recipe']);r['validation_plan_id']=batch.id
                    timeout=min(estimate*2+5,budget.search_remaining,max(2,budget.remaining*.3))
                    if candidate['recipe']['model_family']==automl.FAMILY:timeout=min(timeout,challenger_state['worker_seconds'])
                    rounds=candidate.get('bag_iterations',candidate.get('iterations'))
                    result,predictions=run_candidate({**candidate,'recipe':r},batch,timeout,batch.seed,
                        evaluation_only=True,iterations_override=rounds)
                    result.update(accepted=False,phase='confirmation',confirmation_windows=[w.id for w in windows],
                        hyperparameters_frozen=True,early_stopping_on_controls=False)
                    if result['status']=='ok' and predictions is not None:
                        offset=0;metrics=[]
                        for window in windows:
                            valid=predictions[0][offset:offset+len(window.valid)];offset+=len(window.valid)
                            try:
                                evidence={**result,**evaluate(data,window,(valid,predictions[1]),result),
                                    'predictions':(valid,predictions[1]),'split':window}
                                confirmed.setdefault(candidate['experiment_id'],[]).append(evidence)
                                metrics.append({'split_id':window.id,'metric_value':evidence['metric_value']})
                            except ValueError as error:metrics.append({'split_id':window.id,'error':str(error)})
                        result['control_metrics']=metrics
                    record(result)
            required={window.id for window in control_plan}
            stable=[c for c in ordered if required and {r['split'].id for r in confirmed.get(c['experiment_id'],[])}>=required]
            if data.spec.metric.name in {'f1_macro','balanced_accuracy'}:
                stable=[candidate for candidate in stable if all(
                    row.get('metric_feasibility',{}).get('macro_fold_aggregation_allowed',True)
                    for row in [candidate,*confirmed[candidate['experiment_id']]])]
            common=required if stable else set()
            common_evidence={c['experiment_id']:[r for r in confirmed[c['experiment_id']] if r['split'].id in common] for c in stable}
            if common:
                oof=store_oof(directory/'oof_candidates.npz',data,stable,common_evidence)
                (directory/'oof_manifest.json').write_text(json.dumps(oof,indent=2))
                for candidate in stable:
                    evidence=common_evidence[candidate['experiment_id']]
                    candidate['robust_utility']=robust_utility([candidate['standardized_utility'],*[r['standardized_utility'] for r in evidence]])
                    candidate['confirmation_metrics']=[{'split_id':r['split'].id,'metric_value':r['metric_value']} for r in evidence]
                    candidate['confirmation_metric']=float(np.mean([r['metric_value'] for r in evidence]))
                    rounds=[r.get('iterations') for r in [candidate,*evidence] if r.get('iterations')]
                    if rounds:candidate['refit_iterations']=int(np.median(rounds))
                champion=max(stable,key=lambda c:c['robust_utility']);publish(champion)
                cv_selection={'planned_folds':1+len(control_plan),'completed_candidates':len(stable),
                    'selection_basis':'mean utility minus 0.25 standard deviation on all frozen folds',
                    'candidates':[{'experiment_id':c['experiment_id'],'model_family':c['recipe']['model_family'],
                        'fold_metrics':[c['metric_value'],*[r['metric_value'] for r in common_evidence[c['experiment_id']]]],
                        'mean_metric':float(np.mean([c['metric_value'],*[r['metric_value'] for r in common_evidence[c['experiment_id']]]])),
                        'metric_std':float(np.std([c['metric_value'],*[r['metric_value'] for r in common_evidence[c['experiment_id']]]])),
                        'robust_utility':c['robust_utility']} for c in stable]}
                emit('cross_validation_selected',champion=champion['experiment_id'],folds=1+len(control_plan),candidates=len(stable))
                if len(stable)>=2:
                    stage='ENSEMBLE';blend=choose_blend(data,stable,common_evidence,ensemble_strategy)
                    if blend:
                        checkpoint(data,blend['predictions'][1]);checkpoint_kind='confirmed_champion'
                        emit('ensemble_accepted',weights=blend['weights'],score=blend['metric_value'],confirmation=blend['confirmation_metric'])
                if not blend:
                    multipliers=tune_multipliers(data,champion,common_evidence.get(champion['experiment_id']))
                    if multipliers:
                        checkpoint(data,multipliers['predictions'][1]);checkpoint_kind='confirmed_champion'
                        emit('probability_adjustment_accepted',score=multipliers['metric_value'],confirmation=multipliers['confirmation_metric'])
                if finalization=='fold_average':
                    components=blend['components'] if blend else [champion]
                    weights=blend['weights'] if blend else [1.]
                    predictions=[]
                    for component in components:
                        rows=[component,*common_evidence[component['experiment_id']]]
                        predictions.append(np.mean([row['predictions'][1] for row in rows],axis=0))
                    final=sum(weight*prediction for weight,prediction in zip(weights,predictions))
                    if multipliers:final=adjust_probabilities(final,multipliers['weights'])
                    checkpoint(data,final);fold_bagged_published=True;checkpoint_kind='fold_bagged'
                    emit('fold_bagged_published',components=len(components),folds=1+len(control_plan))
            else:
                cv_selection={'planned_folds':1+len(control_plan),'completed_candidates':0,
                    'selection_basis':'search holdout fallback; confirmation incomplete or class support prevents comparable macro aggregation'}
        # Publish refit only after every component succeeds; previous good file survives.
        if champion and not stop['requested'] and not fold_bagged_published:
            stage='FINALIZE';components=blend['components'] if blend else [champion];weights=blend['weights'] if blend else [1.]
            refitted=[];full_split=Split(np.arange(len(data.y)),np.empty(0,dtype=int),'full_refit','All known labels after selection',seed)
            for c in components:
                estimate=max(2,c['runtime_sec']*1.6)
                if budget.remaining<max(1.5,estimate*.45):break
                timeout=max(1,min(estimate*2+5,budget.remaining-min(5,budget.reserve*.25)))
                if c['recipe']['model_family']==automl.FAMILY:timeout=min(timeout,challenger_state['worker_seconds']*1.8)
                result,predictions=run_candidate(c,full_split,timeout,seed,refit=True,iterations_override=c.get('bag_iterations',c.get('refit_iterations',c.get('iterations'))))
                result['accepted']=False;result['phase']='refit'
                record(result)
                if result['status']!='ok' or predictions is None:break
                refitted.append(predictions[1])
            if len(refitted)==len(components):
                final=sum(w*p for w,p in zip(weights,refitted))
                if multipliers:final=adjust_probabilities(final,multipliers['weights'])
                checkpoint(data,final);refit_published=True;checkpoint_kind='full_refit';emit('full_refit_published',components=len(components))
        if not champion and not stop['requested'] and not any(row['status']=='ok' for row in history):
            fatal_error={'exception_class':'NoSuccessfulTraining','message':'No worker completed training; the retained CSV is only an emergency constant fallback'}
            emit('training_unavailable',failed_experiments=len(history),status='fatal_with_fallback')
            return 1
        return 0
    except Exception as error:
        # The valid atomic output exists. Keep a diagnostic and exit gracefully.
        emit('orchestrator_error',exception_class=type(error).__name__,message=str(error)[:500])
        fatal_error={'exception_class':type(error).__name__,'message':str(error)[:500]}
        (directory/'error.txt').write_text(traceback.format_exc())
        return 1
    finally:
        summary={'runtime_code_sha256':source_digest,'mode':actual_mode,'budget_seconds':budget.total,'elapsed_seconds':time.monotonic()-start,'first_checkpoint_latency':first_latency,'validation':split.json(),'metric':data.spec.metric.name,'dummy_utility':initial_utility,'experiments':len(history),'successful_experiments':sum(r['status']=='ok' for r in history),'accepted_experiments':sum(r.get('accepted',False) for r in history),'champion_metric':champion['metric_value'] if champion else None,'champion_recipe':champion['recipe'] if champion else None,'confirmation_metric':champion.get('confirmation_metric') if champion else None,'ensemble':{k:v for k,v in blend.items() if k not in {'components','predictions'}} if blend else None,'multipliers':{k:v for k,v in multipliers.items() if k!='predictions'} if multipliers else None,'qwen_calls':qwen.requests,'qwen_proposals':qwen.calls,'qwen_tokens':qwen.tokens,'qwen_events':qwen.events,'max_live_worker_threads':max((r.get('max_live_threads',0) for r in history),default=0),'peak_worker_memory_mb':max((r['peak_memory_mb'] for r in history),default=0),'cancelled':stop['requested'],'output':str(inputs.output),'output_sha256':hashlib.sha256(inputs.output.read_bytes()).hexdigest() if inputs.output.exists() else None}
        summary['full_refit_published']=refit_published
        summary.update(status='fatal_with_fallback' if fatal_error and inputs.output.exists() else 'fatal' if fatal_error else checkpoint_kind,
            terminal_phase=stage,checkpoint_kind=checkpoint_kind,trained_output=checkpoint_kind!='valid_dummy',
            fit_success=sum(r['status']=='ok' for r in history),
            search_fit_success=sum(r['status']=='ok' and not r.get('phase') for r in history),
            full_refit_done=refit_published,fatal_exception=fatal_error,
            platform_deadline_seconds=10800,platform_safety_margin_seconds=max(0,10800-budget.total))
        if challenger_candidate:
            challenger_state.update(confirmation_folds=len(confirmed.get(challenger_candidate['experiment_id'],[])),
                selected=any(c['recipe']['model_family']==automl.FAMILY for c in (blend['components'] if blend else [champion] if champion else [])))
        summary['automl_challenger']=challenger_state
        summary['adaptive_search']=scheduler.summary()
        summary['cross_validation']=cv_selection
        summary['confirmation_plan']=[s.json() for s in control_plan]
        summary['validation_guards']={'sealed_before_search':bool(control_plan),
            'search_pool_rows':len(np.union1d(split.train,split.valid)),'confirmation_fits_per_candidate':control_fits,
            'validation_windows':len(control_plan),'control_labels_used_for_early_stopping':False,
            'target_profile_scope':data.spec.profile.get('target_scope','Search-only task without controls')}
        summary['confirmation_metrics']=champion.get('confirmation_metrics',[]) if champion else []
        summary['feature_group_trials']=sum(r.get('feature_trial',{}).get('kind')=='feature_group' for r in history)
        summary['feature_ablation_checks']=sum(r.get('feature_trial',{}).get('kind')=='feature_ablation' for r in history)
        summary['accepted_feature_ablations']=sum(r.get('accepted',False) and r.get('feature_trial',{}).get('kind')=='feature_ablation' for r in history)
        summary.update(researcher_provider=research_provider,researcher_model=qwen.model if qwen.configured else None,researcher_calls=qwen.requests,researcher_proposals=qwen.calls,researcher_tokens=qwen.tokens,researcher_events=qwen.events,
            ensemble_strategy=ensemble_strategy,finalization_requested=finalization,fold_bagged_done=fold_bagged_published)
        if research_provider!='organizer-qwen':
            summary.update(qwen_calls=0,qwen_proposals=0,qwen_tokens=0,qwen_events=[])
        final_recipe={'output_sha256':summary['output_sha256'],'full_refit':refit_published,'seed':seed,'labels':data.spec.labels,'components':[{'recipe':c['recipe'],'iterations':c.get('refit_iterations',c.get('iterations')),'experiment_id':c['experiment_id'],**({'challenger_policy':c['challenger_policy']} if 'challenger_policy' in c else {})} for c in (blend['components'] if blend else [champion] if champion else [])],'blend_weights':blend['weights'] if blend else [1.],'probability_multipliers':multipliers['weights'] if multipliers else None}
        (directory/'final_recipe.json').write_text(json.dumps(final_recipe,ensure_ascii=False,indent=2,allow_nan=False))
        (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        emit('DONE',summary=str(directory/'summary.json'),elapsed=round(summary['elapsed_seconds'],3),champion_metric=summary['champion_metric'])
