import numpy as np
import pytest
from atlas.metrics import Metric,infer_metric
from atlas.features import SparseText
from atlas.ensemble import multiclass_decision_weights,adjust_probabilities


def test_mape_inference_and_zero_policy_are_explicit():
    assert infer_metric('Primary metric: MAPE')[0]=='mape'
    assert Metric('mape').direction==-1
    assert Metric('mape').score([2.,4.],[1.,5.])==pytest.approx(.375)
    assert Metric('mape').score([0.,2.],[0.,1.])==pytest.approx(.25)


def test_pr_auc_is_not_roc_auc_or_average_precision():
    assert infer_metric('Primary metric: PR-AUC')[0]=='pr_auc'
    assert infer_metric('Primary metric: average precision')[0]=='average_precision'
    y=np.array([0,1,0,1]);p=np.array([.9,.8,.7,.6]);p=np.column_stack([1-p,p])
    pr=Metric('pr_auc',2).score(y,p)
    assert pr!=pytest.approx(Metric('average_precision',2).score(y,p))
    assert pr!=pytest.approx(Metric('roc_auc',2).score(y,p))


@pytest.mark.parametrize('mode',['word','char','word_char'])
def test_sparse_text_fits_only_training_vocabulary_and_handles_empty(mode):
    encoder=SparseText(mode).fit(['alpha alpha','beta beta','alpha beta'])
    shape=encoder.transform(['unseenunicode']).shape
    assert shape[0]==1 and shape[1]>0
    assert all('unseenunicode' not in v.vocabulary_ for v in encoder.vectorizers)
    empty=SparseText(mode).fit(['','',None])
    assert empty.transform(['unknown']).shape==(1,1)
    assert empty.transform(['unknown']).nnz==0


def test_multiclass_coordinates_improve_supported_primary_predictions():
    y=np.repeat(np.arange(3),10)
    p=np.vstack([np.tile([.7,.2,.1],(10,1)),np.tile([.4,.35,.25],(10,1)),np.tile([.1,.2,.7],(10,1))])
    metric=Metric('f1_macro',3)
    weights=multiclass_decision_weights(y,p,metric)
    assert weights is not None
    assert metric.score(y,adjust_probabilities(p,weights))>metric.score(y,p)
    assert np.all((weights>=.67)&(weights<=1.5))
    assert multiclass_decision_weights(y[:12],p[:12],metric) is None


def test_csv_probability_validation_checks_bounds_and_normalization(tmp_path):
    from scripts.synthetic import create
    from atlas.contract import read_inputs
    from atlas.task import load
    from atlas.submit import validate_frame
    data=load(read_inputs(create(tmp_path,'logloss')))
    frame=data.sample.copy();frame['common']=.2;frame['rare']=.2
    with pytest.raises(ValueError,match='sum to one'):validate_frame(data,frame)
    frame['common']=-.2;frame['rare']=1.2
    with pytest.raises(ValueError,match='outside'):validate_frame(data,frame)


def test_missing_macro_classes_are_marked_incomparable(tmp_path):
    from scripts.synthetic import create
    from atlas.contract import read_inputs
    from atlas.task import load
    from atlas.judge import evaluate
    from atlas.validation import Split
    data=load(read_inputs(create(tmp_path,'integer_labels')))
    indices=np.flatnonzero(data.y==0)
    split=Split(np.setdiff1d(np.arange(len(data.y)),indices),indices,'stratified','support check',42)
    prediction=np.zeros((len(indices),len(data.spec.labels)));prediction[:,0]=1.
    result=evaluate(data,split,(prediction,np.empty((0,len(data.spec.labels)))))
    assert result['metric_feasibility']['unvalidated_class_indices']
    assert not result['metric_feasibility']['macro_fold_aggregation_allowed']
