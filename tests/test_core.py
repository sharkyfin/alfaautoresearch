import json,os,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as sk
from atlas.contract import read_inputs,confined
from atlas.task import load
from atlas.metrics import Metric,UnsupportedTask,infer_metric
from atlas.validation import design
from atlas.features import FeatureBuilder,MatrixBuilder
from atlas.recipes import validate,RecipeError,compile_recipe,recipe_hash
from atlas.submit import checkpoint,dummy_prediction,validate_frame
from atlas.models import fit_predict
from scripts.synthetic import create,SCENARIOS

@pytest.mark.parametrize('scenario',SCENARIOS)
def test_scenario_output_or_explicit_unsupported(tmp_path,scenario):
    task=create(tmp_path/scenario,scenario)
    if scenario in ['conflict','multiple_targets']:
        with pytest.raises(UnsupportedTask):load(read_inputs(task))
        return
    data=load(read_inputs(task));split=design(data)
    p=dummy_prediction(data.y,data.spec,len(data.test));checkpoint(data,p)
    frame=pd.read_csv(data.inputs.output,dtype={'id':'string'},keep_default_na=False)
    validate_frame(data,frame)
    if scenario=='future':
        assert split.kind=='time'
        t=pd.to_datetime(data.train.event_date)
        assert t.iloc[split.train].max()<t.iloc[split.valid].min()
    elif scenario=='group':
        assert split.kind=='group'
        assert not set(data.train.entity_id.iloc[split.train])&set(data.train.entity_id.iloc[split.valid])
    elif scenario=='datetime_iid':assert split.kind=='random'
    elif scenario=='duplicates':
        assert split.kind=='duplicate_group'
        a=pd.util.hash_pandas_object(data.train[data.spec.features],index=False)
        assert not set(a.iloc[split.train])&set(a.iloc[split.valid])
    elif scenario=='singleton':assert 0 in split.train and 0 not in split.valid
    elif scenario=='unknown_metric':assert any('UNSUPPORTED_METRIC' in w for w in data.spec.warnings)

@pytest.mark.parametrize('name',['rmse','mae','mse','r2','rmsle'])
def test_regression_metrics(name):
    y=np.array([1.,2.,3.,6]);p=np.array([1.1,2.7,2.9,5.2]);m=Metric(name)
    expected={'rmse':np.sqrt(sk.mean_squared_error(y,p)),'mse':sk.mean_squared_error(y,p),'mae':sk.mean_absolute_error(y,p),'r2':sk.r2_score(y,p),'rmsle':np.sqrt(sk.mean_squared_log_error(y,p))}[name]
    assert m.score(y,p)==pytest.approx(expected)
    assert (m.utility(expected)>m.utility(expected+.1))==(name!='r2')

@pytest.mark.parametrize('name',['f1_macro','f1_weighted','f1_micro','accuracy','balanced_accuracy','logloss','roc_auc'])
def test_classification_metrics(name):
    y=np.array([0,0,0,1,1,2]);p=np.array([[.8,.1,.1],[.7,.2,.1],[.2,.5,.3],[.2,.6,.2],[.1,.3,.6],[.1,.2,.7]])
    m=Metric(name,3);hard=p.argmax(1)
    if name.startswith('f1'):expected=sk.f1_score(y,hard,labels=[0,1,2],average=name.removeprefix('f1_'),zero_division=0)
    elif name=='accuracy':expected=sk.accuracy_score(y,hard)
    elif name=='balanced_accuracy':expected=sk.balanced_accuracy_score(y,hard)
    elif name=='logloss':expected=sk.log_loss(y,p,labels=[0,1,2])
    else:expected=sk.roc_auc_score(y,p,multi_class='ovr',average='macro')
    assert m.score(y,p)==pytest.approx(expected)

def test_macro_fixed_classset_and_primary():
    p=np.array([[.8,.1,.1],[.8,.1,.1]])
    assert Metric('f1_macro',3).score([0,0],p)==pytest.approx(1/3)
    assert infer_metric('Основная метрика — Macro F1.\nWeighted F1 and Accuracy are auxiliary.')[0]=='f1_macro'
    assert infer_metric('Primary metric: MAE\nRMSE reference')[0]=='mae'

def test_fold_fitted_features(tmp_path):
    data=load(read_inputs(create(tmp_path)));spec=data.spec
    a=data.train.iloc[:100].copy();b=data.train.iloc[100:].copy();a['category']='a';b['category']='unseen'
    f=FeatureBuilder(spec,[{'op':'frequency','columns':['category']}]).fit(a)
    assert (f.transform(b)['__atlas_op_0_category']==0).all()
    assert 'unseen' not in f.counts['category']
    matrix=MatrixBuilder('linear',f.cat,f.num).fit(f.transform(a));assert np.isfinite(matrix.transform(f.transform(b)).toarray()).all() if hasattr(matrix.transform(f.transform(b)),'toarray') else True

@pytest.mark.parametrize('family',['linear','hist','extra_trees','catboost','lightgbm','xgboost'])
@pytest.mark.parametrize('scenario',['iid','integer_labels'])
def test_real_model_apis(tmp_path,family,scenario):
    data=load(read_inputs(create(tmp_path,scenario)));split=design(data)
    pv,pt,info=fit_predict(data,{'model_family':family,'params':{'iterations':25} if family!='linear' else {},'feature_ops':[]},split.train,split.valid,threads=1)
    assert np.isfinite(data.spec.metric.score(data.y[split.valid],pv));checkpoint(data,pt)
    if data.spec.classification:assert pt.shape[1]==len(data.spec.labels)

@pytest.mark.parametrize('mutation',[{'model_family':'pretrained'},{'params':{'eval_metric':'accuracy'}},{'feature_ops':[{'op':'ratio','columns':['response','x0']}]},{'operation':'shell'},{'params':{'depth':'7'}},{'feature_ops':[{'op':'exec','columns':['x0']}]},{'target_transform':'__import__'},{'max_runtime_sec':-1},{'validation_plan_id':'other'}])
def test_recipe_rejects_unsafe(tmp_path,mutation):
    data=load(read_inputs(create(tmp_path)));data.spec.validation_plan=design(data).json()
    raw={'model_family':'catboost','params':{},'max_runtime_sec':10,**mutation}
    with pytest.raises(RecipeError):validate(raw,data.spec)

def test_atomic_writer_survives_invalid(tmp_path):
    data=load(read_inputs(create(tmp_path)));p=dummy_prediction(data.y,data.spec,len(data.test));checkpoint(data,p);old=data.inputs.output.read_bytes()
    with pytest.raises(ValueError):checkpoint(data,np.full(len(data.test),np.nan))
    assert data.inputs.output.read_bytes()==old

def test_path_injection(tmp_path):
    with pytest.raises(ValueError):confined(tmp_path,'../secret')

def test_unknown_primary_does_not_optimize_auxiliary():
    assert infer_metric('Primary metric: mystery(beta=2)\nAccuracy is auxiliary')[0] is None

@pytest.mark.parametrize('family',['linear','hist','extra_trees','catboost','lightgbm','xgboost'])
def test_binary_model_api(tmp_path,family):
    data=load(read_inputs(create(tmp_path,'binary_imbalance')));s=design(data)
    p,t,info=fit_predict(data,{'model_family':family,'params':{'iterations':25} if family!='linear' else {}},s.train,s.valid,threads=1)
    assert t.shape==(len(data.test),2)
    assert np.isfinite(data.spec.metric.score(data.y[s.valid],p))

def test_clip_fitted_only_training_and_text_vectorizer(tmp_path):
    data=load(read_inputs(create(tmp_path,'text')))
    a=data.train.iloc[:100].copy();b=data.train.iloc[100:].copy();a.x0=1.;b.x0=1e5
    f=FeatureBuilder(data.spec,[{'op':'clip','columns':['x0']}]).fit(a)
    assert (f.transform(b)['__atlas_op_0_x0']==1).all()
    m=MatrixBuilder('linear',f.cat,f.num,data.spec.text_candidates).fit(f.transform(a))
    assert any(name.startswith('text_') for name,_,_ in m.encoder.transformers)
    assert m.transform(f.transform(b)).shape[0]==len(b)

def test_negated_forecast_and_gap(tmp_path):
    task=create(tmp_path/'ordinary','datetime_iid');task.write_text(task.read_text()+'\nThis is not forecasting future periods.\n')
    data=load(read_inputs(task));assert design(data).kind=='random'
    task=create(tmp_path/'forecast','future');task.write_text(task.read_text()+'\ngap_rows: 5\n')
    data=load(read_inputs(task));split=design(data)
    assert split.kind=='time' and split.valid.min()-split.train.max()==6

def test_extreme_finite_dummy(tmp_path):
    data=load(read_inputs(create(tmp_path)));data.y=np.full(len(data.y),1e308)
    p=dummy_prediction(data.y,data.spec,len(data.test));assert np.isfinite(p).all();checkpoint(data,p)

@pytest.mark.parametrize('mutation',[{'model_family':[]},{'feature_ops':[{'op':[],'columns':['x0']}]},{'params':{'weight':[]}},{'target_transform':[]},{'parent_experiment':{}},{'id':False}])
def test_recipe_malformed_types_are_classified(tmp_path,mutation):
    data=load(read_inputs(create(tmp_path)));data.spec.validation_plan=design(data).json()
    with pytest.raises(RecipeError):validate({'model_family':'hist',**mutation},data.spec)

def test_reproducibility(tmp_path):
    data=load(read_inputs(create(tmp_path,'integer_labels')));s=design(data,seed=7)
    recipe={'model_family':'extra_trees','params':{'iterations':25},'feature_ops':[]}
    a=fit_predict(data,recipe,s.train,s.valid,seed=11,threads=1)
    b=fit_predict(data,recipe,s.train,s.valid,seed=11,threads=1)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])

def test_malicious_column_is_data_and_compiler_literal(tmp_path):
    task=create(tmp_path);name="x'); __import__('os').system('false'); #"
    for filename in ['train.csv','test.csv']:
        frame=pd.read_csv(tmp_path/filename,dtype={'id':'string'});frame.rename(columns={'x0':name}).to_csv(tmp_path/filename,index=False)
    dictionary=pd.read_csv(tmp_path/'data_dictionary.csv');dictionary.loc[dictionary.feature=='x0','feature']=name;dictionary.to_csv(tmp_path/'data_dictionary.csv',index=False)
    data=load(read_inputs(task));s=design(data);data.spec.validation_plan=s.json()
    r=validate({'model_family':'hist','params':{'iterations':25},'feature_ops':[{'op':'product','columns':[name,'x1']}],'max_runtime_sec':10},data.spec)
    source=compile_recipe(r,tmp_path/'job.json',Path(__file__).resolve().parents[1]);compile(source,'recipe','exec')
    p,t,i=fit_predict(data,r,s.train,s.valid,threads=1);assert np.isfinite(p).all();checkpoint(data,t)
