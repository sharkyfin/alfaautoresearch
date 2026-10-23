import pytest
from scripts.resume_frozen_final import remaining_keys

def row(case,status='trained_success'):
 return dict(case=case,seed=42,arm='candidate',mode='live-agent',status=status)

def test_completed_failures_are_preserved_and_never_retried():
 keys=[('a',42,'candidate','live-agent'),('b',42,'candidate','live-agent'),('c',42,'candidate','live-agent')]
 assert remaining_keys(keys,[row('a'),row('b','runtime_failed')])=={keys[2]}

def test_duplicate_completed_key_is_rejected():
 with pytest.raises(ValueError,match='Duplicate'):remaining_keys([('a',42,'candidate','live-agent')],[row('a'),row('a')])

def test_unknown_trial_key_is_rejected():
 with pytest.raises(ValueError,match='outside'):remaining_keys([('a',42,'candidate','live-agent')],[row('b')])
