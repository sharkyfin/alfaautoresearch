"""In-memory, seeded TPE studies with isolated feature/target representations."""
from __future__ import annotations

import copy
import hashlib
import json
import math

from .recipes import recipe_hash


DEFAULTS={
    'lightgbm':{'num_leaves':31,'min_child_samples':20,'reg_lambda':0.,'learning_rate':.045,'colsample_bytree':1.,'subsample':1.},
    'xgboost':{'max_depth':6,'min_child_weight':1.,'reg_lambda':1.,'learning_rate':.05,'colsample_bytree':1.,'subsample':1.},
    'hist':{'max_leaf_nodes':31,'l2_regularization':0.,'min_samples_leaf':20,'learning_rate':.08},
    'extra_trees':{'min_samples_leaf':2,'max_features':.8},
    'linear':{'alpha':10.,'C':1.}}


def study_key(recipe):
    core={key:recipe.get(key,[] if key.startswith('feature_') else 'none') for key in
          ['model_family','feature_ops','feature_expressions','target_transform']}
    # Explanations do not change the measured representation. Different training
    # caps have separate studies so a longer fit is not mistaken for a better HPO point.
    core['feature_expressions']=[{key:item[key] for key in ['name','expression']} for item in core['feature_expressions']]
    core['iterations']=recipe.get('params',{}).get('iterations',500) if recipe.get('model_family')!='linear' else None
    return hashlib.sha256(json.dumps(core,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


class AdaptiveTPE:
    def __init__(self,spec,seed=42):
        self.spec=spec;self.seed=seed;self.studies={};self.records=[];self.pending={}
        self.completed=0;self.proposals=0;self.family_proposals={}
        self.enabled='optuna' in spec.available_capabilities

    def space(self,family):
        from optuna.distributions import CategoricalDistribution as Cat,FloatDistribution as Float,IntDistribution as Int
        if family=='lightgbm':
            out={'num_leaves':Int(7,127),'min_child_samples':Int(5,150),
                 'reg_lambda':Float(0,50),'learning_rate':Float(.015,.15,log=True),
                 'colsample_bytree':Float(.6,1.),'subsample':Float(.65,1.)}
        elif family=='xgboost':
            out={'max_depth':Int(3,9),'min_child_weight':Float(1.,30.,log=True),
                 'reg_lambda':Float(0,50),'learning_rate':Float(.015,.15,log=True),
                 'colsample_bytree':Float(.6,1.),'subsample':Float(.65,1.)}
        elif family=='hist':
            out={'max_leaf_nodes':Int(7,63),'min_samples_leaf':Int(5,100),
                 'l2_regularization':Float(0,30),'learning_rate':Float(.02,.15,log=True)}
        elif family=='extra_trees':
            out={'min_samples_leaf':Int(1,15),'max_features':Float(.4,1.)}
        else:
            out={'C' if self.spec.classification else 'alpha':Float(.001,1000,log=True)}
        if self.spec.classification and family!='linear':out['weight']=Cat(['none','sqrt','balanced'])
        if self.spec.text_candidates and family!='linear':out['text_svd_components']=Cat([0,32,64,128])
        return out

    def values(self,recipe,space):
        defaults={**DEFAULTS[recipe['model_family']],'weight':'none','text_svd_components':64}
        values={name:recipe.get('params',{}).get(name,defaults[name]) for name in space}
        try:return values if all(space[name]._contains(space[name].to_internal_repr(value)) for name,value in values.items()) else None
        except (ValueError,TypeError):return None

    def get_study(self,recipe):
        import optuna
        key=study_key(recipe)
        if key not in self.studies:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler=optuna.samplers.TPESampler(seed=(self.seed+int(key[:8],16))%2**32,n_startup_trials=4,n_ei_candidates=16)
            study=optuna.create_study(direction='maximize',sampler=sampler)
            space=self.space(recipe['model_family'])
            self.studies[key]=(study,space)
            for row in self.records:
                if study_key(row['recipe'])==key:self.add_observation(study,space,row)
        return self.studies[key]

    def add_observation(self,study,space,result):
        import optuna
        values=self.values(result['recipe'],space)
        utility=result.get('standardized_utility')
        if values is None or utility is None or not math.isfinite(utility):return
        study.add_trial(optuna.trial.create_trial(params=values,distributions=space,value=float(utility)))

    def abandon(self):
        if not self.pending:return
        import optuna
        for study,trial in self.pending.values():study.tell(trial,state=optuna.trial.TrialState.FAIL)
        self.pending.clear()

    def propose(self,candidate,seen=()):
        if not self.enabled:return None
        self.abandon()
        from optuna.distributions import CategoricalDistribution,IntDistribution
        import optuna
        parent=candidate['recipe'];study,space=self.get_study(parent)
        for _ in range(8):
            trial=study.ask()
            params={}
            for name,distribution in space.items():
                if isinstance(distribution,CategoricalDistribution):value=trial.suggest_categorical(name,distribution.choices)
                elif isinstance(distribution,IntDistribution):value=trial.suggest_int(name,distribution.low,distribution.high,log=distribution.log,step=distribution.step)
                else:value=trial.suggest_float(name,distribution.low,distribution.high,log=distribution.log,step=distribution.step)
                params[name]=value
            raw=copy.deepcopy(parent)
            raw['params']={**raw.get('params',{}),**params}
            raw.update(id='tpe_'+str(self.proposals+1),parent_experiment=candidate['experiment_id'],
                       statement='Seeded TPE trial from measured results for this model and feature representation')
            raw.pop('validation_plan_id',None)
            digest=recipe_hash(raw)
            if digest in seen:
                study.tell(trial,state=optuna.trial.TrialState.FAIL);continue
            self.pending[digest]=(study,trial)
            self.proposals+=1
            family=raw['model_family'];self.family_proposals[family]=self.family_proposals.get(family,0)+1
            return raw
        return None

    def observe(self,result):
        if not self.enabled:return
        digest=recipe_hash(result['recipe'])
        pending=self.pending.pop(digest,None)
        success=result.get('status')=='ok' and result.get('standardized_utility') is not None
        if pending:
            import optuna
            study,trial=pending
            study.tell(trial,float(result['standardized_utility']) if success else None,
                       state=optuna.trial.TrialState.COMPLETE if success else optuna.trial.TrialState.FAIL)
            if success:self.completed+=1
        if success:
            row={key:copy.deepcopy(result[key]) for key in ['recipe','standardized_utility']}
            self.records.append(row)
            study=self.studies.get(study_key(result['recipe']))
            if study is not None and pending is None:self.add_observation(*study,row)

    def summary(self):
        return {'backend':'optuna-tpe' if self.enabled else 'bounded-local-search',
                'studies':len(self.studies),'proposed_trials':self.proposals,
                'completed_trials':self.completed,'family_proposals':self.family_proposals}
