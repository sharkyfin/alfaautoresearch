"""Task discovery using training schema, dictionary, sample and description."""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
import re
import importlib.metadata as metadata
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .contract import Inputs
from .metrics import Metric,REGRESSION,CLASSIFICATION,UnsupportedTask,infer_metric

APPROVED={'numpy':'1.26.4','pandas':'2.2.3','scipy':'1.12.0','scikit-learn':'1.5.2','lightgbm':'4.6.0','xgboost':'2.1.4'}

def capabilities(strict=False):
    result={}
    for name,version in APPROVED.items():
        try: installed=metadata.version(name)
        except metadata.PackageNotFoundError: continue
        if not strict or installed==version: result[name]=installed
    return result

def read_csv(path,dtype=None):
    return pd.read_csv(path,keep_default_na=False,na_values=[''],dtype=dtype,low_memory=False)

@dataclass
class TaskSpec:
    target: str
    ids: list[str]
    features: list[str]
    problem_type: str
    metric: Metric
    labels: list
    prediction_kind: str
    probability_columns: dict[str,int]
    numeric_features: list[str]
    categorical_features: list[str]
    datetime_candidates: list[str]
    text_candidates: list[str]
    group_candidates: list[str]
    available_capabilities: dict
    warnings: list[str]=field(default_factory=list)
    suspicious_leakage_features: list[str]=field(default_factory=list)
    validation_plan: dict=field(default_factory=dict)
    profile: dict=field(default_factory=dict)

    @property
    def classification(self): return self.problem_type!='regression'
    def json(self): return asdict(self)

@dataclass
class Dataset:
    inputs: Inputs
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    dictionary: pd.DataFrame
    spec: TaskSpec
    y: np.ndarray


def load(inputs: Inputs, strict=False) -> Dataset:
    dictionary=read_csv(inputs.dictionary) if inputs.dictionary else pd.DataFrame()
    role={}
    if {'feature','role'}<=set(dictionary.columns): role={str(r.feature):str(r.role).lower() for r in dictionary.itertuples()}
    ids_hint=[c for c,r in role.items() if r in {'id','identifier','идентификатор'}]
    train=read_csv(inputs.train,dtype={c:'string' for c in ids_hint})
    test=read_csv(inputs.test,dtype={c:'string' for c in ids_hint})
    sample=read_csv(inputs.sample,dtype={c:'string' for c in ids_hint})
    missing=[c for c in train if c not in test]
    declared=[c for c,r in role.items() if r in {'target','label','целевая переменная','таргет'}]
    sample_targets=[c for c in missing if c in sample]
    warnings=[]
    if declared:
        if len(declared)!=1 or declared[0] not in missing: raise UnsupportedTask(f'Dictionary target conflicts with train/test schema: {declared}, {missing}')
        target=declared[0]
        if sample_targets and sample_targets!=[target]: raise UnsupportedTask('Sample and dictionary disagree on target')
    elif len(sample_targets)==1: target=sample_targets[0]
    elif len(missing)==1: target=missing[0]
    else: raise UnsupportedTask(f'Ambiguous or multi-target schema: {missing}')
    if len(sample)!=len(test): raise UnsupportedTask('Sample and test row count differ')
    if train[target].isna().any():
        warnings.append('Dropped rows with missing training target'); train=train.loc[train[target].notna()].reset_index(drop=True)
    if not len(train): raise UnsupportedTask('No known training targets')
    shared=[c for c in sample if c in test and c!=target]
    ids=[c for c in shared if c in ids_hint] if ids_hint else [c for c in shared if test[c].is_unique]
    for c in ids:
        # Re-read IDs as strings, preserving leading zeros and uint64 identifiers.
        for frame,path in ((train,inputs.train),(test,inputs.test),(sample,inputs.sample)):
            col=read_csv(path,dtype={c:'string'})[c]
            if len(col)==len(frame): frame[c]=col
    if ids:
        if test.duplicated(ids).any() or sample.duplicated(ids).any(): raise UnsupportedTask('Submission IDs must be unique')
        a=set(map(tuple,test[ids].astype(str).to_numpy()));b=set(map(tuple,sample[ids].astype(str).to_numpy()))
        if a!=b: raise UnsupportedTask('Sample and test IDs differ')
    metric_name,metric_kwargs=infer_metric(inputs.description)
    desc=inputs.description.lower()
    classification=(metric_name in CLASSIFICATION or (metric_name is None and (not pd.api.types.is_numeric_dtype(train[target]) or re.search(r'classification|классификац',desc))))
    if metric_name is None:
        explicit_metric=re.search(r'(?:^|\n)\s*(?:primary metric|metric|метрика)\s*:\s*([^\n]+)',desc)
        if explicit_metric: warnings.append('UNSUPPORTED_METRIC: '+explicit_metric.group(1))
        metric_name='f1_macro' if classification else 'rmse'
        warnings.append('Metric not recognized; explicit fallback '+metric_name)
    labels=[]
    if classification:
        enc=LabelEncoder(); y=enc.fit_transform(train[target]);labels=enc.classes_.tolist()
    else:
        y=pd.to_numeric(train[target],errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(y).all(): raise UnsupportedTask('Nonfinite regression target')
    if metric_name=='rmsle' and (y<0).any(): raise UnsupportedTask('RMSLE has negative training targets')
    positive=1
    positive_match=re.search(r'(?:positive_label|positive class|положительный класс)\s*[:=]\s*[`"\x27]?([^`"\x27\s,]+)',desc)
    if positive_match and labels:
        matches=[i for i,l in enumerate(labels) if str(l).lower()==positive_match.group(1)]
        if not matches: raise UnsupportedTask('Unknown positive class')
        positive=matches[0]
    metric=Metric(metric_name,len(labels),positive,metric_kwargs)
    kind='label' if classification else 'value';proba_cols={}
    out=[c for c in sample if c not in ids]
    class_probability_schema=classification and len(labels)>1 and len(out)==len(labels) and set(out)=={str(label) for label in labels}
    if classification and (metric.needs_proba or class_probability_schema):
        kind='probability'
        if len(out)==1 and len(labels)==2: proba_cols[out[0]]=positive
        elif len(out)==len(labels):
            for c in out:
                indices=[i for i,l in enumerate(labels) if str(l)==c]
                if len(indices)!=1: raise UnsupportedTask('Probability headers must match class labels exactly')
                proba_cols[c]=indices[0]
        else: raise UnsupportedTask('Ambiguous probability output schema')
    elif out!=[target]: raise UnsupportedTask('Unsupported multi-output sample schema')
    extra_targets=set(missing)-{target}
    features=[c for c in test if c in train and c not in ids and c not in extra_targets and role.get(c) not in {'ignore','exclude','target','label'}]
    if not features: warnings.append('No usable features: dummy-only')
    numeric=[];cats=[];dates=[];texts=[];groups=[]
    for c in features:
        s=train[c];r=role.get(c,'')
        if r=='group' or (re.search(r'(^|_)(entity|patient|building|user|group|customer)(_id)?$',c.lower()) and s.nunique()<len(s)*.8): groups.append(c)
        hinted=str(dictionary.loc[dictionary['feature']==c,'dtype'].iloc[0]).lower() if {'feature','dtype'}<=set(dictionary) and (dictionary['feature']==c).any() else ''
        if hinted in {'datetime','date','timestamp'} or (not pd.api.types.is_numeric_dtype(s) and re.search(r'date|time|дата',c,re.I) and s.dropna().astype(str).str.contains(r'[-/:]').mean()>.8): dates.append(c)
        elif hinted in {'category','categorical','string'}: cats.append(c)
        elif pd.api.types.is_numeric_dtype(s) and hinted not in {'category','categorical','string'}: numeric.append(c)
        elif hinted in {'numeric','float','integer'} or pd.to_numeric(s.astype('string').str.replace(',','.',regex=False),errors='coerce').notna().mean()>.95: numeric.append(c)
        else:
            cats.append(c)
        if c in cats and s.dropna().astype(str).str.len().median()>35: texts.append(c)
    spec=TaskSpec(target,ids,features,'binary' if len(labels)==2 else 'multiclass' if classification else 'regression',metric,labels,kind,proba_cols,numeric,cats,dates,texts,groups,capabilities(strict),warnings)
    spec.profile=profile(train,test,spec)
    return Dataset(inputs,train,test,sample,dictionary,spec,np.asarray(y))


def column_profile(train,test,spec,c):
    s=train[c];t=test[c]
    info={'dtype':str(s.dtype),'missing':round(float(s.isna().mean()),4),'cardinality':int(s.nunique())}
    if c in spec.numeric_features:
        def numeric_values(series):
            if not pd.api.types.is_numeric_dtype(series):series=series.astype('string').str.replace(' ','',regex=False).str.replace(',','.',regex=False)
            return pd.to_numeric(series,errors='coerce').replace([np.inf,-np.inf],np.nan)
        a=numeric_values(s);test_numeric=numeric_values(t)
        scale=float(a.std()) if a.notna().sum()>1 else 0.
        info['median_shift_std']=float((test_numeric.median()-a.median())/scale) if scale>0 and np.isfinite(scale) and a.notna().any() and test_numeric.notna().any() else 0.
        info['zero_fraction']=float((a==0).mean())
        info['nonfinite_or_missing_fraction']=float(a.isna().mean())
        info['quantiles']={str(k):float(v) for k,v in a.quantile([0,.25,.5,.75,1]).items() if np.isfinite(v)}
    elif c not in spec.datetime_candidates:
        info['unseen_fraction']=float((~t.isin(s.dropna())).mean()) if len(t) else 0.
        lengths=s.dropna().astype(str).str.len()
        info['median_length']=float(lengths.median()) if len(lengths) else 0.
    return info


def profile(train,test,spec):
    cols={c:column_profile(train,test,spec,c) for c in spec.features[:200]}
    y=train[spec.target]
    target={'counts':{str(k):int(v) for k,v in y.value_counts().items()}} if spec.classification else {'quantiles':{str(k):float(v) for k,v in y.quantile([0,.01,.25,.5,.75,.99,1]).items()}}
    return {'train_shape':list(train.shape),'test_shape':list(test.shape),'columns':cols,'target':target,'memory_mb':float(train.memory_usage(deep=True).sum()/2**20)}


def leakage_signals(data,train_indices):
    """Signal direct target duplicates using only the fixed training fold."""
    if len(train_indices)<10:return []
    sample=data.train.iloc[np.asarray(train_indices)[:5000]]
    target=sample[data.spec.target].to_numpy(dtype=object)
    return [c for c in data.spec.features[:300] if sample[c].notna().all() and np.array_equal(sample[c].to_numpy(dtype=object),target)]
