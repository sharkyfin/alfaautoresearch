"""The unseen panel must reject protocol/scorer drift before data access."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.benchmark_suite import final_protocol, validate_final_freeze


def example():
    args=SimpleNamespace(seeds='42,53',budget=120.0,threads=8,memory_mb=8192,max_calls=2,
        exclude_families='',exclude_by_arm='',recipe_by_arm='',blend_by_arm='',finalization_by_arm='')
    protocol=final_protocol(args,['live-agent'])
    values=dict(cases=['a','b'],archives={'baseline':'old-sha','candidate':'new-sha'},
        protocol=protocol,manifest_sha='manifest-sha',evaluator_hashes={'scorer.py':'scorer-sha'})
    freeze={'cases':['b','a'],'archives':values['archives'],'protocol':deepcopy(protocol),
        'manifest_sha256':values['manifest_sha'],'evaluator_fingerprints':values['evaluator_hashes']}
    return freeze,values


def test_final_freeze_accepts_identical_protocol_independent_of_case_order():
    freeze,values=example()
    validate_final_freeze(freeze,**values)


def test_final_freeze_rejects_budget_seed_mode_and_intervention_changes():
    freeze,values=example()
    for key,changed in [('budget_seconds',180),('seeds',[71]),('modes',['deterministic']),
                        ('threads',4),('memory_mb',4096),('max_qwen_calls',3),
                        ('exclude_families','catboost'),('recipe_by_arm','candidate=tuned.json')]:
        altered=deepcopy(freeze);altered['protocol'][key]=changed
        with pytest.raises(ValueError,match='protocol'):
            validate_final_freeze(altered,**values)


def test_final_freeze_rejects_dataset_archive_and_scorer_changes():
    freeze,values=example()
    for key,changed in [('cases',['a']),('archives',{'candidate':'another-sha'}),
                        ('manifest_sha256','another-manifest'),
                        ('evaluator_fingerprints',{'scorer.py':'another-scorer'})]:
        altered=deepcopy(freeze);altered[key]=changed
        with pytest.raises(ValueError,match=key):
            validate_final_freeze(altered,**values)
