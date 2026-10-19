"""Fold-fitted preprocessing and bounded semantic feature operations."""
from __future__ import annotations
import ipaddress
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def numeric(series):
    if pd.api.types.is_numeric_dtype(series): return pd.to_numeric(series,errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)
    s=series.astype('string').str.strip().str.replace(' ','',regex=False).str.replace(',','.',regex=False)
    return pd.to_numeric(s,errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)

class FeatureBuilder:
    def __init__(self,spec,ops=(),feature_function=None):
        self.spec=spec;self.ops=ops;self.feature_function=feature_function
    def fit(self,x):
        self.counts={c:x[c].fillna('__MISSING__').astype(str).value_counts(normalize=True).to_dict() for c in self.spec.categorical_features}
        self.clip_limits={c:tuple(numeric(x[c]).quantile([.01,.99]).fillna(0)) for op in self.ops if op['op']=='clip' for c in op['columns']}
        self.group_stats={}
        for i,op in enumerate(self.ops):
            if op['op']=='group_mean':
                tmp=pd.DataFrame({'g':x[op['columns'][0]].fillna('__MISSING__').astype(str),'v':numeric(x[op['columns'][1]])})
                self.group_stats[i]=tmp.groupby('g')['v'].mean().to_dict()
        z=self.transform(x)
        self.columns=list(z.columns);self.cat=[c for c in z if not pd.api.types.is_numeric_dtype(z[c])]
        self.num=[c for c in z if c not in self.cat]
        return self
    def transform(self,x):
        out={}
        for c in self.spec.features:
            if c in self.spec.numeric_features: out[c]=numeric(x[c])
            elif c in self.spec.datetime_candidates:
                d=pd.to_datetime(x[c],errors='coerce',utc=True)
                out[c+'__epoch']=d.astype('int64').where(d.notna(),np.nan)/1e9
                for attr in ['year','month','day','dayofweek','hour']:
                    out[c+'__'+attr]=getattr(d.dt,attr).astype(float)
                for attr,period in [('month',12),('dayofweek',7),('hour',24)]:
                    out[c+'__'+attr+'_sin']=np.sin(2*np.pi*getattr(d.dt,attr)/period)
                    out[c+'__'+attr+'_cos']=np.cos(2*np.pi*getattr(d.dt,attr)/period)
            else:
                s=x[c].fillna('__MISSING__').astype(str).str.slice(0,4096)
                out[c]=s
                if c in self.spec.text_candidates:
                    out[c+'__length']=s.str.len().astype(float);out[c+'__words']=s.str.count(r'\S+').astype(float)
        for i,op in enumerate(self.ops):
            columns=op.get('columns',[]);key=f'__atlas_op_{i}';kind=op['op']
            if self.feature_function is not None and kind not in {'drop','ip','clip','combine'}: continue
            if kind=='drop':
                for c in columns: out.pop(c,None)
            elif kind=='clip':
                for c in columns: out[key+'_'+c]=numeric(x[c]).clip(*self.clip_limits[c])
            elif kind=='combine':
                out[key]=x[columns[0]].fillna('__MISSING__').astype(str)+'::'+x[columns[1]].fillna('__MISSING__').astype(str)
            elif kind=='frequency':
                for c in columns: out[key+'_'+c]=x[c].fillna('__MISSING__').astype(str).map(self.counts.get(c,{})).fillna(0).astype(float)
            elif kind=='text_stats':
                for c in columns:
                    s=x[c].fillna('').astype(str);out[key+'_'+c+'_len']=s.str.len().astype(float);out[key+'_'+c+'_digits']=s.str.count(r'\d').astype(float)
            elif kind=='ip':
                for c in columns:
                    def encode(v):
                        try:return int(ipaddress.ip_address(str(v)))
                        except ValueError:return np.nan
                    values=x[c].map(encode).astype(float)
                    out[key+'_'+c]=values;out[key+'_'+c+'_prefix']=np.floor(values/65536)
            elif kind=='group_mean':
                out[key]=x[columns[0]].fillna('__MISSING__').astype(str).map(self.group_stats[i]).astype(float)
            else:
                a=numeric(x[columns[0]])
                if kind=='log1p': out[key]=np.sign(a)*np.log1p(np.abs(a))
                elif kind=='sqrt': out[key]=np.sign(a)*np.sqrt(np.abs(a))
                elif kind=='square':out[key]=np.clip(a,-1e100,1e100)**2
                elif kind=='ratio':
                    b=numeric(x[columns[1]]);out[key]=(a/b.where(b.abs()>1e-9)).clip(-1e12,1e12)
                elif kind=='difference':out[key]=a-numeric(x[columns[1]])
                elif kind=='product':out[key]=(a.clip(-1e100,1e100)*numeric(x[columns[1]]).clip(-1e100,1e100)).clip(-1e200,1e200)
        if self.feature_function is not None: out.update(self.feature_function(x,self))
        z=pd.DataFrame(out,index=x.index)
        if not len(z.columns):z['__constant__']=0.
        for c in z.select_dtypes(include='number'): z[c]=z[c].replace([np.inf,-np.inf],np.nan).clip(-1e30,1e30)
        return z

class MatrixBuilder:
    def __init__(self,kind,cat,num,text=()):self.kind=kind;self.cat=cat;self.num=num;self.text=[c for c in text if c in cat][:2]
    def fit(self,z):
        if self.kind=='linear':
            transformers=[]
            if self.num:transformers.append(('n',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True)),('scale',StandardScaler())]),self.num))
            ordinary=[c for c in self.cat if c not in self.text]
            if ordinary:transformers.append(('c',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=2,max_categories=200,sparse_output=True),ordinary))
            for i,c in enumerate(self.text):transformers.append((f'text_{i}',TfidfVectorizer(max_features=12000,ngram_range=(1,2),sublinear_tf=True,strip_accents='unicode',token_pattern=r'(?u)\b\w+\b'),c))
            self.encoder=ColumnTransformer(transformers,sparse_threshold=1.0).fit(z)
        else:
            self.maps={c:{v:i for i,v in enumerate(z[c].drop_duplicates())} for c in self.cat}
            self.medians={c:float(z[c].median()) if z[c].notna().any() else 0. for c in self.num}
        return self
    def transform(self,z):
        if self.kind=='linear':return self.encoder.transform(z)
        if self.kind=='catboost':
            z=z.copy()
            for c in self.num:z[c]=z[c].fillna(self.medians[c]).clip(-1e30,1e30)
            return z
        z=z.copy()
        for c in self.cat:
            codes=z[c].map(self.maps[c]).fillna(-1).astype(int)
            if self.kind=='lightgbm':z[c]=pd.Categorical.from_codes(codes.to_numpy(),categories=list(range(len(self.maps[c]))))
            else:z[c]=codes.astype(float)
        if self.kind!='lightgbm':
            for c in self.num:z[c]=z[c].fillna(self.medians[c]).clip(-1e30,1e30)
        return z
