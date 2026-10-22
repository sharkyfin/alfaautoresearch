"""Lean cost-aware portfolio, local search and evidence-guided mock researcher."""
from __future__ import annotations
import copy
import numpy as np
from .recipes import PARAMS,RecipeError,recipe_hash,merge_hypothesis_features,validate
from .hpo import AdaptiveTPE,study_key
from .branches import preprocessing_candidates,transfer_candidates
from .judge import meaningful_gain


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
        self.initial_attempts=0;self.family_best={};self.hpo=AdaptiveTPE(spec,seed);self.stop_reason=None
        self.nodes={};self.closed_keys=set();self.focus_key=None;self.pending_branch=None
        self.last_research_step=0;self.branch_research_due=False;self.request_researcher=True
        self.branches=[{'id':0,'kind':'portfolio','status':'running','experiments':0,'hpo_trials':0,'no_gain':0}]
    def observe(self,result):
        completed=self.hpo.completed
        self.hpo.observe(result)
        branch=self.branches[-1];branch['experiments']+=1
        branch['hpo_trials']+=self.hpo.completed-completed
        if result.get('status')!='ok' or result.get('standardized_utility') is None:
            branch['no_gain']+=1;return
        key=study_key(result['recipe']);old_node=self.nodes.get(key)
        better=old_node is None or meaningful_gain(result['standardized_utility'],old_node['standardized_utility'])
        record={name:copy.deepcopy(result[name]) for name in
                ['experiment_id','recipe','standardized_utility','metric_value','runtime_sec','iterations','diagnostics'] if name in result}
        record.setdefault('metric_value',result['standardized_utility']*self.spec.metric.direction)
        if old_node is None or result['standardized_utility']>old_node['standardized_utility']:self.nodes[key]=record
        focus=self.nodes.get(self.focus_key)
        follows_feature=(self.focus_key is not None and key!=self.focus_key and
                         result.get('search_stage') in {'RESEARCH','FEATURE_ABLATION'} and
                         (focus is None or meaningful_gain(result['standardized_utility'],focus['standardized_utility'])))
        if follows_feature:
            self.focus_key=key;branch['representation']=key
        improved=(key==self.focus_key and better) if self.focus_key is not None else result.get('accepted',False)
        branch['no_gain']=0 if improved else branch['no_gain']+1
        branch['best_utility']=max(branch.get('best_utility',float('-inf')),result['standardized_utility'])
        family=result['recipe']['model_family'];old=self.family_best.get(family)
        if old is None or result['standardized_utility']>old['standardized_utility']:
            self.family_best[family]=record
    def should_restart(self,no_gain):
        branch=self.branches[-1]
        stalled=branch['no_gain'] if self.focus_key is not None else no_gain
        trials=branch['hpo_trials'] if self.focus_key is not None else self.hpo.completed
        return stalled>=18 and trials>=12 if self.hpo.enabled else stalled>=10
    def should_stop(self,no_gain):
        # A plateau ends a branch. Only exhaustion/deadline ends the search.
        return self.stop_reason is not None
    def parent(self,champion):
        return self.nodes.get(self.focus_key,champion)
    def restart(self,champion):
        if self.focus_key is not None:self.closed_keys.add(self.focus_key)
        self.branches[-1]['status']='stagnated'
        self.hpo.abandon()
        if self.budget.search_remaining<=2:
            self.stop_reason='Research deadline reached';return None
        ranked=sorted(self.nodes.values(),key=lambda r:-r['standardized_utility'])
        donors=ranked[:3];recipients=sorted(self.family_best.values(),key=lambda r:-r['standardized_utility'])
        parents=list({study_key(p['recipe']):p for p in [champion,*ranked[:3]]}.values())
        groups={
            'feature_transfer':((raw,detail) for raw,detail in transfer_candidates(donors,recipients)),
            'preprocessing':((raw,{'operation':kind,'parent_experiment':parent['experiment_id']})
                             for parent in parents for kind,raw in preprocessing_candidates(self.spec,parent)),
            'model_tuning':((p['recipe'],{'parent_experiment':p['experiment_id']}) for p in ranked)}
        kinds=list(groups);offset=(len(self.branches)-1)%len(kinds)
        for kind in kinds[offset:]+kinds[:offset]:
            for raw,detail in groups[kind]:
                family=raw['model_family']
                if self.failures.get(family,0)>=2:continue
                raw=copy.deepcopy(raw)
                available=self.budget.search_remaining
                raw['max_runtime_sec']=min(raw.get('max_runtime_sec',300),max(1,available-1))
                try:clean=validate(raw,self.spec,remaining=available)
                except RecipeError:continue
                key=study_key(clean)
                if key in self.closed_keys:continue
                known=self.nodes.get(key)
                if known is None and recipe_hash(clean) in self.seen:continue
                self.focus_key=key
                self.pending_branch=None if known else (kind.upper(),clean)
                self.branch_research_due=True
                branch={'id':len(self.branches),'kind':kind,'status':'running','model_family':family,
                        'representation':key,'experiments':0,'hpo_trials':0,'no_gain':0,**detail}
                self.branches.append(branch)
                return branch
        self.stop_reason='No untested model, preprocessing or feature-transfer branches remain'
        return None
    def reject(self):
        self.branches[-1]['no_gain']+=1
    def summary(self):
        return {**self.hpo.summary(),'stop_reason':self.stop_reason,'research_steps':self.research_index,
                'research_trial_limit':self.budget.research_trial_limit,'branches_started':len(self.branches),
                'branches_completed':sum(b['status']=='stagnated' for b in self.branches),
                'branch_history':copy.deepcopy(self.branches)}
    def next(self,champion):
        if self.pending_branch is not None:
            self.stage,raw=self.pending_branch;self.pending_branch=None
            self.research_index+=1
            if self.research_index>self.budget.research_trial_limit:
                self.stop_reason='Research trial safety cap reached';return None
            return raw
        if self.focus_key is not None and self.focus_key not in self.nodes:
            if not self.restart(champion):return None
            return self.next(champion)
        early_research=self.mode=='live-agent' and champion is not None and self.initial_attempts>=2 and self.research_index==0
        if self.queue and not early_research:
            self.stage='MODEL_DIVERSITY';self.initial_attempts+=1;return self.queue.pop(0)
        self.stage='RESEARCH';self.research_index+=1
        if not champion or self.research_index>self.budget.research_trial_limit:
            self.stop_reason='No champion or bounded research trial cap reached';return None
        parent=self.parent(champion)
        spec=self.spec;base=copy.deepcopy(parent['recipe']);base.pop('validation_plan_id',None);base['parent_experiment']=parent['experiment_id'];base['id']=f'local_{self.research_index}'
        family=base['model_family'];i=self.research_index
        self.request_researcher=(self.budget.total<300 and i<9 or i==1 or self.branch_research_due or i-self.last_research_step>=12)
        if self.request_researcher and self.mode=='live-agent':
            self.last_research_step=i;self.branch_research_due=False
            if i>=9 or self.focus_key is not None:
                for _,raw in preprocessing_candidates(spec,parent):
                    try:validate(raw,spec,remaining=max(300,self.budget.search_remaining),seen=self.seen)
                    except RecipeError:continue
                    return raw
        growth=None
        branch_step=self.branches[-1]['experiments']+1 if self.focus_key is not None else i
        if branch_step in {3,9,15} and family in {'hist','lightgbm','xgboost'}:
            rounds=base['params'].get('iterations',500)
            maximum=PARAMS[family]['iterations'][1]
            fit_cost=parent.get('runtime_sec',1.)
            if parent.get('iterations',0)>=rounds*.95 and rounds<maximum and self.budget.can_run(fit_cost*2.4,refit_estimate=fit_cost*2.4):
                proposal=copy.deepcopy(base)
                proposal['params']['iterations']=min(maximum,rounds*2)
                if recipe_hash(proposal) not in self.seen:growth=proposal
        due=self.mode=='live-agent' and self.request_researcher
        if growth is None and (i>=9 or self.focus_key is not None) and not due and self.hpo.enabled and self.family_best:
            ranked=[parent] if self.focus_key is not None else sorted(self.family_best.values(),key=lambda c:-c['standardized_utility'])[:2]
            # Give the second family regular opportunities even after a narrow
            # early winner. Its own measured features/transform stay attached.
            index=1 if len(ranked)>1 and i%3==0 else 0
            raw=self.hpo.propose(ranked[index],self.seen)
            if raw is not None:
                self.stage='HYPERPARAMETER_SEARCH';return raw
        # Test semantic/weighting changes before expensive hyperparameter variants.
        if growth is not None:
            base=growth;base['statement']='Training reached its iteration cap; test more rounds within measured fit and refit costs'
        elif spec.classification and i==1 and family!='linear':
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
        return merge_hypothesis_features(base,parent['recipe'])


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
