import numpy as np
import pandas as pd
from atlas.encoding import CrossfitTargetEncoder


def test_crossfit_target_encoder_excludes_own_target_and_unseen_uses_train_prior():
    frame=pd.DataFrame({'category':['a','a','b','b','c','c','d','d','e','e']})
    y=np.arange(10,dtype=float)
    encoder=CrossfitTargetEncoder(seed=17)
    encoded=encoder.fit_transform(frame,y)
    changed=y.copy();changed[0]=10000
    altered=CrossfitTargetEncoder(seed=17).fit_transform(frame,changed)
    assert encoded[0,0]==altered[0,0]
    assert encoder.transform(pd.DataFrame({'category':['unseen']}))[0,0]==np.mean(y)
    assert encoded.shape==(10,1) and np.isfinite(encoded).all()


def test_group_crossfit_excludes_whole_entity_targets():
    frame=pd.DataFrame({'category':['shared']*12})
    groups=np.repeat(np.arange(6),2);y=np.arange(12,dtype=float)
    encoded=CrossfitTargetEncoder(seed=42).fit_transform(frame,y,groups)
    changed=y.copy();changed[:2]+=1000
    altered=CrossfitTargetEncoder(seed=42).fit_transform(frame,changed,groups)
    assert np.array_equal(encoded[:2],altered[:2])


def test_multiclass_crossfit_keeps_global_class_alignment():
    frame=pd.DataFrame({'category':['a','b','c']*6});y=np.tile(np.arange(3),6)
    encoder=CrossfitTargetEncoder(3,seed=4)
    encoded=encoder.fit_transform(frame,y)
    assert encoded.shape==(18,3)
    assert np.allclose(encoded.sum(1),1.)
    assert np.allclose(encoder.transform(pd.DataFrame({'category':['unknown']})),[[1/3]*3])
