"""Lean cost-aware portfolio, local search and evidence-guided mock researcher."""
from __future__ import annotations
import copy
import numpy as np
from .recipes import recipe_hash,merge_hypothesis_features


def recipe(family,params=None,ops=None,statement='',transform='none'):
    return {'model_family':family,'params':params or {},'feature_ops':ops or [],'target_transform':transform,'statement':statement,'max_runtime_sec':300}


def initial_portfolio(spec,budget=10800):
    rounds=150 if budget<120 else 500 if budget<900 else 1800
    out=[recipe('linear',{'C':1} if spec.classification else {'alpha':10},statement='Fast regularized linear baseline')]
    if 'lightgbm' in spec.available_capabilities:out.append(recipe('lightgbm',{'iterations':rounds,'num_leaves':31,'min_child_samples':30},statement='Fast leaf-wise booster with categorical handling'))
    out.append(recipe('hist',{'iterations':min(350,rounds)},statement='Histogram boosting diversity'))
    out.append(recipe('extra_trees',{'iterations':min(250,rounds),'min_samples_leaf':2},statement='Randomized trees for complementary errors'))
    if 'xgboost' in spec.available_capabilities:out.append(recipe('xgboost',{'iterations':rounds,'max_depth':6},statement='Regularized histogram booster diversity'))
    if spec.classification and spec.metric.name in {'f1','f1_macro','balanced_accuracy'} and 'lightgbm' in spec.available_capabilities:
        counts=list(spec.profile.get('target',{}).get('counts',{}).values())
        if counts and max(counts)/max(min(counts),1)>=3:
            out.insert(2,recipe('lightgbm',{'iterations':rounds,'num_leaves':31,'min_child_samples':30,'weight':'sqrt','stopping_metric':'primary'},statement='Measured minority-weighted booster with early stopping on the actual primary metric'))
            position=next(i for i,r in enumerate(out) if r['model_family']=='extra_trees')+1
            ops=[{'op':'frequency','columns':spec.categorical_features[:8]}] if spec.categorical_features else []
            out.insert(position,recipe('extra_trees',{'iterations':min(250,rounds),'min_samples_leaf':2,'weight':'sqrt'},ops,statement='Weighted randomized trees with fold-fitted category rarity'))
    return out

def retain_finalists(candidates,limit=3):
    unique={c['experiment_id']:c for c in candidates}
    ranked=sorted(unique.values(),key=lambda c:-c['standardized_utility'])
    if len(ranked)<=limit:return ranked
    chosen=ranked[:2];best=ranked[0]
    if chosen[0]['recipe']['model_family']==chosen[1]['recipe']['model_family']:
        distinct=next((c for c in ranked[2:] if c['recipe']['model_family']!=best['recipe']['model_family'] and best['standardized_utility']-c['standardized_utility']<=max(.02,abs(best['standardized_utility'])*.05)),None)
        if distinct:chosen.append(distinct)
    for c in ranked:
        if len(chosen)>=limit:break
        if c['experiment_id'] not in {x['experiment_id'] for x in chosen}:chosen.append(c)
    return chosen

class SearchController:
    def __init__(self,spec,budget,mode='deterministic',seed=42):
        self.spec=spec;self.budget=budget;self.mode=mode;self.rng=np.random.default_rng(seed);self.stage='INITIALIZE';self.seen=set();self.failures={};self.research_index=0
        self.queue=initial_portfolio(spec,budget.total)
        self.initial_attempts=0
    def next(self,champion):
        early_research=self.mode=='live-agent' and champion is not None and self.initial_attempts>=2 and self.research_index==0
        if self.queue and not early_research:
            self.stage='MODEL_DIVERSITY';self.initial_attempts+=1;return self.queue.pop(0)
        self.stage='RESEARCH';self.research_index+=1
        if not champion or self.research_index>24:return None
        spec=self.spec;base=copy.deepcopy(champion['recipe']);base.pop('validation_plan_id',None);base['parent_experiment']=champion['experiment_id'];base['id']=f'local_{self.research_index}'
        family=base['model_family'];i=self.research_index
        # Test semantic/weighting changes before expensive hyperparameter variants.
        if spec.classification and i==1 and family!='linear':
            base['params']['weight']='sqrt';base['statement']='Minority recall may improve with sqrt-balanced sample weights'
        elif not spec.classification and i==1 and min(spec.profile['target']['quantiles'].values())>=0:
            base['target_transform']='log1p';base['statement']='Skewed nonnegative target may benefit from log training, scored on original scale'
        elif i in {2,3} and spec.categorical_features:
            base['feature_ops']=[{'op':'frequency','columns':spec.categorical_features[:8]}];base['statement']='Fold-fitted category frequency may expose rarity without target leakage'
            if i==3:base['params']['weight']='balanced' if spec.classification else 'none'
        elif i in {4,5} and len(spec.numeric_features)>=2:
            cols=spec.numeric_features[:2];base['feature_ops']=[{'op':'ratio' if i==4 else 'product','columns':cols}];base['statement']='Bounded numeric interaction may capture nonlinear rates'
        elif i==6 and spec.numeric_features:
            base['feature_ops']=[{'op':'clip','columns':spec.numeric_features[:8]}];base['statement']='Fold-fitted clipped copies may reduce outlier sensitivity while retaining originals'
        elif i==7 and len(spec.categorical_features)>=2:
            base['feature_ops']=[{'op':'combine','columns':spec.categorical_features[:2]}];base['statement']='Categorical composition may expose conditional patterns'
        elif i==8 and spec.categorical_features and spec.numeric_features:
            base['feature_ops']=[{'op':'group_mean','columns':[spec.categorical_features[0],spec.numeric_features[0]]}];base['statement']='Training-fold feature aggregate may capture group scale without target encoding'
        else:
            base['statement']='Focused local hyperparameter search around measured champion'
            if family=='lightgbm':base['params'].update(num_leaves=int(self.rng.choice([15,31,63,127])),min_child_samples=int(self.rng.choice([10,30,60,100])),reg_lambda=float(self.rng.choice([0,1,5,20])))
            elif family=='xgboost':base['params'].update(max_depth=int(self.rng.choice([3,5,7,9])),min_child_weight=int(self.rng.choice([1,5,15])))
            elif family=='extra_trees':base['params'].update(min_samples_leaf=int(self.rng.choice([1,3,5,10])),max_features=float(self.rng.choice([.5,.8,1.])))
            elif family=='hist':base['params'].update(max_leaf_nodes=int(self.rng.choice([15,31,63])),l2_regularization=float(self.rng.choice([0,1,10])))
            else:base['params'].update({'C':float(self.rng.choice([.01,.1,10,100]))} if spec.classification else {'alpha':float(self.rng.choice([.1,1,100,1000]))})
        return merge_hypothesis_features(base,champion['recipe'])


def mock_hypothesis(spec,champion,number):
    base=copy.deepcopy(champion['recipe']);base['id']=f'mock_hyp_{number}';base['parent_experiment']=champion['experiment_id']
    if spec.classification:
        base['params']['weight']='sqrt' if number%2 else 'balanced';base['statement']='Mock researcher: class imbalance evidence suggests increasing minority influence'
    elif len(spec.numeric_features)>=2:
        base['feature_ops']=[{'op':'ratio' if number%2 else 'product','columns':spec.numeric_features[:2]}];base['statement']='Mock researcher: test an explicit interaction against unchanged split'
    elif spec.categorical_features:
        base['feature_ops']=[{'op':'frequency','columns':spec.categorical_features[:4]}];base['statement']='Mock researcher: fold-fitted rarity hypothesis'
    return base


def estimate_runtime(recipe,history,timeout,total_budget):
    measured=[r for r in history if r.get('status')=='ok' and not r.get('refit') and r['recipe']['model_family']==recipe['model_family']]
    if not measured:return min(timeout,max(3.,total_budget*.03))
    previous=measured[-1];old=previous['recipe']['params'];new=recipe.get('params',{})
    scale=max(.5,min(3.,new.get('iterations',500)/max(old.get('iterations',500),1)))
    if recipe['model_family']=='xgboost':
        key='max_depth'
        scale*=2**((new.get(key,7)-old.get(key,7))*.25)
    return min(timeout,max(1.,previous['runtime_sec']*scale*1.2))
