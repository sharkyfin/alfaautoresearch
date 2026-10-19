"""Immutable, common split plan. Never select a split by its model score."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit,train_test_split
from .task import Dataset

@dataclass(frozen=True)
class Split:
    train: np.ndarray
    valid: np.ndarray
    kind: str
    reason: str
    seed: int
    @property
    def id(self):
        return hashlib.sha256(self.train.tobytes()+self.valid.tobytes()+self.kind.encode()).hexdigest()[:16]
    def json(self): return {'id':self.id,'kind':self.kind,'reason':self.reason,'seed':self.seed,'train_rows':len(self.train),'valid_rows':len(self.valid)}


def design(data: Dataset,seed=42,fraction=.2) -> Split:
    train=data.train;spec=data.spec;n=len(train);idx=np.arange(n)
    if n<4: return Split(idx,idx[:0],'unsupported_tiny','Too few rows for honest validation',seed)
    desc=data.inputs.description.lower()
    desc=re.sub(r'[^\n.!?]*(?:not (?:a )?forecast|не прогноз|не является[^\n.!?]*прогноз)[^\n.!?]*[.!?]?', '', desc)
    explicit_time=bool(re.search(r'forecast|forward validation|time.series|прогноз.*будущ|временн.*ряд|будущ.*период|validation\s*:\s*time',desc))
    for col in spec.datetime_candidates:
        dates=pd.to_datetime(train[col],errors='coerce',utc=True)
        testdates=pd.to_datetime(data.test[col],errors='coerce',utc=True)
        if explicit_time and dates.notna().mean()>.95:
            # Split at a timestamp boundary; no equal timestamps straddle the boundary.
            cutoff=dates.sort_values().iloc[max(1,int(n*(1-fraction)))]
            a=idx[(dates<cutoff).to_numpy()];b=idx[(dates>=cutoff).to_numpy()]
            gap_match=re.search(r'gap_rows\s*[:=]\s*(\d+)',desc)
            if gap_match:
                gap=min(int(gap_match.group(1)),len(a)//3)
                if gap:a=a[np.argsort(dates.iloc[a].to_numpy())][:-gap]
            if len(a)>1 and len(b):return Split(a,b,'time',f'Explicit forecasting intent; forward boundary in {col}',seed)
    for col in spec.group_candidates:
        groups=train[col].fillna('__MISSING__').astype(str)
        unseen=(~data.test[col].fillna('__MISSING__').astype(str).isin(groups)).mean()
        explicit=bool(re.search(r'validation\s*:\s*group|unseen (?:entities|groups)|новы[ех].*(?:групп|сущност|пациент)',desc))
        if groups.nunique()>2 and groups.nunique()<n*.8 and (unseen>.8 or explicit):
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(idx,groups=groups))
            return Split(a,b,'group',f'Repeated {col}; new test entity fraction {unseen:.3f}',seed)
    # Exact duplicate feature rows cannot leak across random folds.
    if spec.features:
        fingerprints=pd.util.hash_pandas_object(train[spec.features],index=False).to_numpy()
        if len(np.unique(fingerprints))<n*.98 and len(np.unique(fingerprints))>2:
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(idx,groups=fingerprints))
            return Split(a,b,'duplicate_group','Exact feature duplicates held together',seed)
    if spec.classification:
        counts=np.bincount(data.y,minlength=len(spec.labels))
        rare=counts[data.y]<2
        eligible=idx[~rare]
        if len(eligible)>=4:
            k=len(np.unique(data.y[eligible]));nv=min(len(eligible)-k,max(k,int(len(eligible)*fraction)))
            if nv>=k and len(eligible)-nv>=k:
                a,b=train_test_split(eligible,test_size=nv,stratify=data.y[eligible],random_state=seed)
                a=np.concatenate([a,idx[rare]])
                return Split(a,b,'stratified','Singleton classes retained only in training' if rare.any() else 'IID classification',seed)
        rng=np.random.default_rng(seed);eligible=rng.permutation(eligible)
        b=eligible[:max(1,int(len(eligible)*fraction))]
        a=np.setdiff1d(idx,b)
        return Split(a,b,'rare_degraded','Too few examples for stratification',seed)
    a,b=train_test_split(idx,test_size=fraction,random_state=seed)
    return Split(a,b,'random','IID regression; no evidence for grouped/forward validation',seed)
