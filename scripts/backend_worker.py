"""Development-only competitive classical references; fixed external evaluator."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from atlas.contract import read_inputs
from atlas.task import load
from atlas.submit import checkpoint,dummy_prediction
from atlas.validation import selection_splits,Split
from atlas.features import FeatureBuilder
from atlas.utils import code_digest
from atlas.challenger import CanonicalScore,aligned_prediction,bounded_model_params

class ReferenceScore:
    def __init__(self,metric,log_target=False):self.metric=metric;self.log_target=log_target
    def __call__(self,y,p,**kwargs):
        y=np.asarray(y);p=np.asarray(p)
        if self.log_target:y=np.expm1(y);p=np.maximum(np.expm1(np.clip(p,-30,50)),0.)
        return CanonicalScore(self.metric)(y,p)

def run_backend(args,started):
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=False)
    data=load(read_inputs(args.task),strict=True);checkpoint(data,dummy_prediction(data.y,data.spec,len(data.test)))
    split,_=selection_splits(data,args.seed);data.spec.validation_plan=split.json();metric=data.spec.metric
    remaining=lambda:max(0.,args.budget-(time.monotonic()-started))
    summary={'runtime_code_sha256':code_digest(),'backend':args.mode,'mode':args.mode,'validation':split.json(),
        'budget_seconds':args.budget,'fit_success':0,'experiments':0,'qwen_calls':0,'qwen_tokens':0,
        'checkpoint_kind':'valid_dummy','full_refit_published':False,'champion_recipe':None,
        'backend_wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        summary['elapsed_seconds']=time.monotonic()-started;summary['output_sha256']=hashlib.sha256(data.inputs.output.read_bytes()).hexdigest()
        (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    save()
    try:
        if args.mode=='catboost':
            from atlas.experiments import Executor,register
            from atlas.judge import evaluate
            executor=Executor(data,root/'experiments',args.threads,args.memory_mb,strict=True)
            recipe={'model_family':'catboost','params':{'iterations':6000,'depth':7,'learning_rate':.05,'l2_leaf_reg':3.},
                'feature_ops':[],'feature_expressions':[],'target_transform':'log1p' if metric.name=='rmsle' else 'none',
                'max_runtime_sec':min(1800.,args.budget*.70),'validation_plan_id':split.id}
            result,pred=executor.run(recipe,split,remaining()*.70,args.seed)
            if result['status']!='ok' or pred is None:raise RuntimeError('CatBoost reference failed: '+str(result.get('message',result['status'])))
            result.update(evaluate(data,split,pred,result));register(root/'experiments.jsonl',result)
            checkpoint(data,pred[1]);summary.update(champion_recipe=recipe,champion_metric=result['metric_value'],fit_success=1,experiments=1,
                checkpoint_kind='trained_checkpoint',status='trained_checkpoint',backend_version='1.2.8')
            save();full=Split(np.arange(len(data.y)),np.empty(0,dtype=int),'full_refit','Reference refit after inner early stopping',args.seed)
            if remaining()>max(5,result['runtime_sec']*.3):
                final,pred=executor.run(recipe,full,remaining()-2,args.seed,refit=True,iterations_override=result['iterations'])
                final['phase']='refit';register(root/'experiments.jsonl',final)
                if final['status']=='ok' and pred is not None:
                    checkpoint(data,pred[1]);summary.update(full_refit_published=True,checkpoint_kind='full_refit',status='full_refit',fit_success=2)
                else:summary['refit_error']=final.get('message',final['status'])
        elif args.mode=='autogluon':
            from autogluon.tabular import TabularPredictor
            from autogluon.features.generators import AutoMLPipelineFeatureGenerator
            from autogluon.core.metrics import make_scorer
            fb=FeatureBuilder(data.spec,[]);a=fb.fit_transform(data.train.iloc[split.train]);b=fb.transform(data.train.iloc[split.valid]);test=fb.transform(data.test)
            log_target=metric.name=='rmsle';ya=np.log1p(data.y[split.train]) if log_target else data.y[split.train]
            yb=np.log1p(data.y[split.valid]) if log_target else data.y[split.valid]
            gen=AutoMLPipelineFeatureGenerator(verbosity=0,enable_raw_text_features=False,enable_text_ngram_features=False)
            gen.fit_transform(a,pd.Series(ya,index=a.index));label='__reference_target__'
            while label in a:label+='x'
            a=a.copy();a[label]=ya;b=b.copy();b[label]=yb
            scorer=make_scorer('canonical_'+metric.name,ReferenceScore(metric,log_target),optimum=0 if metric.direction<0 else 1,
                greater_is_better=metric.direction>0,needs_proba=data.spec.classification)
            hyper={'GBM':[{'num_boost_round':5000,'learning_rate':.05},
                {'extra_trees':True,'num_boost_round':5000}, {'num_leaves':63,'learning_rate':.03,'num_boost_round':5000}],
                'CAT':[{'iterations':3000,'depth':6}],
                'RF':[{'n_estimators':250,'max_leaf_nodes':15000,'random_state':args.seed}],
                'XT':[{'n_estimators':250,'max_leaf_nodes':15000,'random_state':args.seed}]}
            hyper={family:[bounded_model_params(family,p,args.threads) for p in configs] for family,configs in hyper.items()}
            folds=5 if split.kind in {'random','stratified'} else 0
            model=TabularPredictor(label=label,problem_type=data.spec.problem_type,eval_metric=scorer,path=str(root/'models'),verbosity=0,
                learner_kwargs={'random_state':args.seed,'label_count_threshold':1})
            model.fit(a,tuning_data=b,feature_generator=gen,time_limit=max(2,remaining()*.68),hyperparameters=hyper,
                num_cpus=args.threads,num_gpus=0,memory_limit=args.memory_mb/1024*.8,fit_strategy='sequential',
                num_bag_folds=folds,num_stack_levels=0,use_bag_holdout=bool(folds),dynamic_stacking=False,
                ag_args_ensemble={'fold_fitting_strategy':'sequential_local'},calibrate_decision_threshold=False,calibrate=False)
            def predict(frame):
                if data.spec.classification:return aligned_prediction(model.predict_proba(frame,as_multiclass=True),data.spec,len(frame))
                value=np.asarray(model.predict(frame))
                if log_target:value=np.maximum(np.expm1(np.clip(value,-30,50)),0.)
                return aligned_prediction(value,data.spec,len(frame))
            checkpoint(data,predict(test));score=metric.score(data.y[split.valid],predict(b.drop(columns=label)))
            summary.update(champion_recipe={'model_family':'autogluon_classical_reference'},champion_metric=score,
                fit_success=len(model.model_names()),experiments=len(model.model_names()),checkpoint_kind='trained_checkpoint',status='trained_checkpoint',
                backend_version='1.3.1',bag_folds=folds,target_transform='log1p' if log_target else 'none')
            save()
            if remaining()>8:
                try:
                    model.refit_full('best',set_best_to_refit_full=True,num_cpus=args.threads,num_gpus=0,fit_strategy='sequential')
                    checkpoint(data,predict(test));summary.update(full_refit_published=True,checkpoint_kind='full_refit',status='full_refit')
                except Exception as error:summary['refit_error']=type(error).__name__+': '+str(error)[:300]
            summary['best_model']=model.model_best
        else:raise ValueError('Unapproved competitive reference')
        summary['trained_output']=True;save();return 0
    except Exception as error:
        summary.update(status='fatal_with_fallback',error=type(error).__name__+': '+str(error)[:500]);save();raise
