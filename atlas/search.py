"""Lean cost-aware portfolio, local search and evidence-guided mock researcher."""
from __future__ import annotations
import copy
import numpy as np
from .recipes import PARAMS,RecipeError,recipe_hash,merge_hypothesis_features
from .hpo import AdaptiveTPE


def recipe(family,params=None,ops=None,statement='',transform='none'):
    return {'model_family':family,'params':params or {},'feature_ops':ops or [],'target_transform':transform,'statement':statement,'max_runtime_sec':300}


def initial_portfolio(spec,budget=10800):
    rounds=150 if budget<120 else 500 if budget<900 else 1800
    out=[recipe('linear',{'C':1} if spec.classification else {'alpha':10},statement='Fast regularized linear baseline')]
    if spec.classification and spec.text_candidates:
        out.append(recipe('linear',{'C':1,'text_analyzer':'word_char'},
            statement='Probe bounded word and character TF-IDF before an early booster can exclude sparse text'))
    if 'lightgbm' in spec.available_capabilities:out.append(recipe('lightgbm',{'iterations':rounds,'num_leaves':31,'min_child_samples':30},statement='Fast leaf-wise booster with categorical handling'))
    if 'catboost' in spec.available_capabilities:
        out.append(recipe('catboost',{'iterations':rounds,'depth':6,'l2_leaf_reg':3.},statement='Native categorical CPU booster, trained only on this task'))
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
        self.observations=[];self.priority_events=[]
        self.research_limit=(min(384,max(96,int(budget.total/15))) if budget.total>=1800 else min(192,max(48,int(budget.total/30))))
    def observe(self,result):
        self.hpo.observe(result)
        family=result['recipe']['model_family']
        previous=self.family_best.get(family)
        old=previous['standardized_utility'] if previous else None
        value=result.get('standardized_utility')
        self.observations.append({'family':family,'ok':result.get('status')=='ok',
            'seconds':max(.1,result.get('runtime_sec',1.)),
            'normalized_gain':max(0.,(value-old)/max(abs(old),1e-6)) if value is not None and old is not None else 0.})
        if result.get('status')!='ok' or result.get('standardized_utility') is None:return
        family=result['recipe']['model_family'];old=self.family_best.get(family)
        if old is None or result['standardized_utility']>old['standardized_utility']:
            self.family_best[family]={key:copy.deepcopy(result[key]) for key in
                ['experiment_id','recipe','standardized_utility','runtime_sec','iterations'] if key in result}
    def should_stop(self,no_gain):
        limit=18 if self.budget.total<1800 else min(96,48+int(self.budget.total/450))
        explored=len(self.family_best)>=min(3,len(initial_portfolio(self.spec,self.budget.total)))
        if no_gain>=limit and self.hpo.completed>=12 and explored:
            self.stop_reason='TPE search stagnated after exploring multiple model families';return True
        if not self.hpo.enabled and no_gain>=(20 if self.budget.total>=1800 else 10):
            self.stop_reason='Bounded local search stagnated';return True
        return False
    def summary(self):
        return {**self.hpo.summary(),'stop_reason':self.stop_reason,'research_steps':self.research_index,
                'research_limit':self.research_limit,'family_priorities':self.priority_events[-20:],
                'quarantined_families':sorted(f for f,n in self.failures.items() if n>=2)}
    def prioritized_family(self,ranked):
        """Task-local improvement/cost rule; no policy learned from benchmarks."""
        best=ranked[0]['standardized_utility'];scores=[]
        for candidate in ranked:
            family=candidate['recipe']['model_family']
            rows=[row for row in self.observations if row['family']==family]
            cost=float(np.median([row['seconds'] for row in rows])) if rows else 1.
            reliability=(1+sum(row['ok'] for row in rows))/(2+len(rows))
            gap=max(0.,(best-candidate['standardized_utility'])/max(abs(best),1e-6))
            evidence=np.mean([row['normalized_gain'] for row in rows[-5:]]) if rows else 0.
            uncertainty=.015/np.sqrt(1+self.hpo.family_proposals.get(family,0))
            priority=float((evidence+uncertainty)*reliability*np.exp(-min(gap,10.)*4)/max(cost,.5))
            scores.append((priority,candidate))
        priority,chosen=max(scores,key=lambda pair:pair[0])
        if self.research_index>=12 and self.research_index%6==0:
            chosen=min(ranked,key=lambda row:self.hpo.family_proposals.get(row['recipe']['model_family'],0))
            priority=next(value for value,row in scores if row is chosen)
        self.priority_events.append({'step':self.research_index,'family':chosen['recipe']['model_family'],
            'estimated_normalized_gain_per_second':priority,'considered_families':len(ranked)})
        return chosen
    def next(self,champion):
        while self.queue and self.failures.get(self.queue[0]['model_family'],0)>=2:self.queue.pop(0)
        early_research=self.mode=='live-agent' and champion is not None and self.initial_attempts>=2 and self.research_index==0
        if self.queue and not early_research:
            self.stage='MODEL_DIVERSITY';self.initial_attempts+=1;return self.queue.pop(0)
        self.stage='RESEARCH';self.research_index+=1
        if not champion or self.research_index>self.research_limit:
            self.stop_reason='No champion or bounded research trial cap reached';return None
        eligible=sorted((c for f,c in self.family_best.items() if self.failures.get(f,0)<2),
                        key=lambda c:-c['standardized_utility'])
        recovering=self.failures.get(champion['recipe']['model_family'],0)>=2
        if recovering:
            if not eligible:
                self.stop_reason='All observed model families are quarantined';return None
            champion=eligible[0]
        spec=self.spec;base=copy.deepcopy(champion['recipe']);base.pop('validation_plan_id',None);base['parent_experiment']=champion['experiment_id'];base['id']=f'local_{self.research_index}'
        family=base['model_family'];i=self.research_index
        growth=None
        if i==10 and self.budget.total>=1800 and family!='linear' and base['params'].get('seed_bag',1)==1:
            cost=champion.get('runtime_sec',1.)
            if self.budget.can_run(cost*2.5,finalization_estimate=cost*9):
                growth=copy.deepcopy(base);growth['params']['seed_bag']=2
                growth['statement']='Test two independent seeds on this legal fold; retain only measured gains after controls'
        if i in {3,9,15} and family in {'hist','catboost','lightgbm','xgboost'}:
            rounds=base['params'].get('iterations',500)
            maximum=PARAMS[family]['iterations'][1]
            fit_cost=champion.get('runtime_sec',1.)
            if champion.get('iterations',0)>=rounds*.95 and rounds<maximum and self.budget.can_run(fit_cost*2.4,refit_estimate=fit_cost*2.4):
                proposal=copy.deepcopy(base)
                proposal['params']['iterations']=min(maximum,rounds*2)
                if recipe_hash(proposal) not in self.seen:growth=proposal
        if growth is None and (i>=9 or recovering) and self.hpo.enabled and eligible:
            ranked=eligible[:(3 if self.budget.total>=1800 else 2)]
            # Give the second family regular opportunities even after a narrow
            # early winner. Its own measured features/transform stay attached.
            index=1 if len(ranked)>1 and i%3==0 else 0
            parent=self.prioritized_family(ranked) if self.budget.total>=1800 else ranked[index]
            raw=self.hpo.propose(parent,self.seen)
            if raw is not None:
                self.stage='HYPERPARAMETER_SEARCH';return raw
        # Test semantic/weighting changes before expensive hyperparameter variants.
        if growth is not None:
            base=growth
            if base['params'].get('seed_bag',1)==1:base['statement']='Training reached its iteration cap; test more rounds within measured fit and refit costs'
        elif i==1 and spec.text_candidates and family=='linear':
            base['params']['text_analyzer']='word_char';base['statement']='Test bounded character n-grams alongside word TF-IDF on the same training fold'
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
            if family=='catboost':base['params'].update(depth=int(self.rng.choice([4,6,8])),l2_leaf_reg=float(self.rng.choice([1,3,10,30])))
            elif family=='lightgbm':base['params'].update(num_leaves=int(self.rng.choice([15,31,63,127])),min_child_samples=int(self.rng.choice([10,30,60,100])),reg_lambda=float(self.rng.choice([0,1,5,20])))
            elif family=='xgboost':base['params'].update(max_depth=int(self.rng.choice([3,5,7,9])),min_child_weight=int(self.rng.choice([1,5,15])))
            elif family=='extra_trees':base['params'].update(min_samples_leaf=int(self.rng.choice([1,3,5,10])),max_features=float(self.rng.choice([.5,.8,1.])))
            elif family=='hist':base['params'].update(max_leaf_nodes=int(self.rng.choice([15,31,63])),l2_regularization=float(self.rng.choice([0,1,10])))
            else:base['params'].update({'C':float(self.rng.choice([.01,.1,10,100]))} if spec.classification else {'alpha':float(self.rng.choice([.1,1,100,1000]))})
        # Keep the known-good checkpoint, but do not freeze a new experiment
        # back to a model family already blocked by the executor's failure rule.
        if recovering:self.stage='MODEL_DIVERSITY'
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
    scale*=new.get('seed_bag',1)/max(old.get('seed_bag',1),1)
    if recipe['model_family']=='xgboost':
        key='max_depth'
        scale*=2**((new.get(key,7)-old.get(key,7))*.25)
    return min(timeout,max(1.,previous['runtime_sec']*scale*1.2))


STRATEGIC_DIRECTIONS = {'improve_family', 'another_family', 'semantic_features', 'imbalance',
                        'target_transform', 'complementarity', 'compute_allocation'}


def strategic_options(spec, history, champion, remaining):
    """Legal actions with task-local cost evidence; no extra Qwen call or labels."""
    rows = [r for r in history if not r.get('phase')]
    failures = {f: sum(r.get('status') != 'ok' for r in rows if r['recipe']['model_family'] == f) for f in PARAMS}
    families = [f for f in PARAMS if failures[f] < 2 and
                (f not in {'catboost', 'lightgbm', 'xgboost'} or f in spec.available_capabilities)]
    current = champion['recipe']['model_family'] if champion else None
    counts = list(spec.profile.get('target', {}).get('counts', {}).values())
    imbalance = bool(counts and max(counts) / max(1, min(counts)) >= 1.5)
    nonnegative = not spec.classification and min(spec.profile.get('target', {}).get('quantiles', {'0': -1}).values()) >= 0
    directions = ['improve_family', 'semantic_features', 'compute_allocation']
    alternatives = [f for f in families if f != current]
    if alternatives: directions += ['another_family', 'complementarity']
    if spec.classification and imbalance and spec.metric.name in {'f1', 'f1_macro', 'balanced_accuracy'}: directions.append('imbalance')
    if nonnegative: directions.append('target_transform')
    costs = []
    for family in families:
        observations = [r for r in rows if r['recipe']['model_family'] == family]
        times = [r['runtime_sec'] for r in observations if r.get('status') == 'ok']
        costs.append({'family': family, 'tested': len(observations),
                      'accepted': sum(bool(r.get('accepted')) for r in observations),
                      'median_seconds': round(float(np.median(times)), 2) if times else None,
                      'failed': failures[family]})
    return {'directions': directions, 'families': families, 'family_costs': costs,
            'remaining_seconds': round(remaining),
            'efforts': ['quick', 'standard', 'deep'],
            'scope': 'One bounded experiment; no control labels, metric changes or new API calls. '
                     'Complementarity is a hypothesis, confirmed only by existing independent ensemble selection.'}


def resolve_strategy(raw, scheduler, champion, history, fallback, remaining):
    """Compile a direction-only suggestion into the existing typed Recipe DSL."""
    if not isinstance(raw, dict) or 'strategy' not in raw: return raw, None
    action = raw['strategy']
    if not isinstance(action, dict) or set(action) - {'direction', 'reason', 'family', 'effort'}:
        raise RecipeError('Invalid research strategy fields')
    if not isinstance(action.get('reason', ''), str) or len(action.get('reason', '')) > 240:
        raise RecipeError('Research strategy reason must be a bounded string')
    options = strategic_options(scheduler.spec, history, champion, remaining)
    direction = action.get('direction'); effort = action.get('effort', 'standard')
    if not isinstance(direction, str) or direction not in options['directions'] or effort not in options['efforts']:
        raise RecipeError('Research strategy is not eligible for this task')
    parent = champion['recipe']; current = parent['model_family']; family = action.get('family', current)
    if not isinstance(family, str) or family not in options['families']:
        raise RecipeError('Research strategy selected an unavailable or quarantined family')
    provided = 'model_family' in raw
    if provided:
        proposal = copy.deepcopy(raw); proposal.pop('strategy')
        family = proposal['model_family']
        if 'family' in action and action['family'] != family: raise RecipeError('Strategy family contradicts recipe')
    else:
        if set(raw) != {'strategy'}: raise RecipeError('Direction-only action must contain only strategy')
        proposal = copy.deepcopy(parent)
        if direction in {'another_family', 'complementarity'}:
            if 'family' not in action:
                other = [row for row in options['family_costs'] if row['family'] != current]
                family = min(other, key=lambda row: (row['tested'] > 0, row['failed'], row['median_seconds'] or 0))['family']
            previous = scheduler.family_best.get(family)
            base = previous['recipe'] if previous else next(r for r in initial_portfolio(scheduler.spec, scheduler.budget.total) if r['model_family'] == family)
            proposal = copy.deepcopy(base)
            proposal['feature_ops'] = copy.deepcopy(parent.get('feature_ops', []))
            proposal['feature_expressions'] = copy.deepcopy(parent.get('feature_expressions', []))
        elif direction == 'semantic_features':
            raise RecipeError('Semantic direction needs an explicit bounded feature hypothesis')
        elif direction == 'imbalance':
            proposal['params']['weight'] = 'balanced' if parent['params'].get('weight') == 'sqrt' else 'sqrt'
        elif direction == 'target_transform':
            proposal['target_transform'] = 'none' if parent.get('target_transform') == 'log1p' else 'log1p'
        else:
            selected = champion
            if direction == 'compute_allocation':
                candidates = [x for f,x in scheduler.family_best.items() if f in options['families']]
                if candidates: selected = scheduler.prioritized_family(sorted(candidates,key=lambda x:-x['standardized_utility']))
            proposed = scheduler.hpo.propose(selected, scheduler.seen) if scheduler.hpo.enabled else None
            proposal = proposed or copy.deepcopy(fallback)
            family = proposal['model_family']
    from .recipes import validate,merge_hypothesis_features
    validate(merge_hypothesis_features(proposal,parent),scheduler.spec,remaining=max(1800.,remaining))
    if family not in options['families']: raise RecipeError('Strategy family is unavailable or quarantined')
    if direction in {'another_family', 'complementarity'} and family == current:
        raise RecipeError('Diversity strategy must change model family')
    if direction in {'improve_family', 'semantic_features', 'imbalance', 'target_transform'} and family != current:
        raise RecipeError('Strategy must retain the current model family')
    from .research import feature_changes
    from .recipes import merge_hypothesis_features
    changes = feature_changes(parent, merge_hypothesis_features(proposal, parent))
    if direction == 'semantic_features' and not changes: raise RecipeError('Semantic strategy did not change features')
    if direction != 'semantic_features' and changes: raise RecipeError('Model strategies must isolate model changes from features')
    if direction == 'imbalance' and proposal.get('params', {}).get('weight', 'none') == parent['params'].get('weight', 'none'):
        raise RecipeError('Imbalance strategy must test a different legal class weight')
    if direction == 'target_transform' and proposal.get('target_transform', 'none') == parent.get('target_transform', 'none'):
        raise RecipeError('Target strategy must change the target transform')
    measured = [row['median_seconds'] for row in options['family_costs'] if row['family'] == family and row['median_seconds'] is not None]
    estimate = measured[0] if measured else max(3., scheduler.budget.total*.03)
    scale = {'quick': 1.5, 'standard': 3., 'deep': 6.}[effort]
    proposal['max_runtime_sec'] = min(proposal.get('max_runtime_sec', 300), max(3., estimate*scale+3), max(1., remaining))
    return proposal, {**action, 'resolved_family': family, 'estimated_fit_seconds': estimate,
                      'worker_cap_seconds': proposal['max_runtime_sec']}
