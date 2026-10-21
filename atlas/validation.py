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
    time_column: str | None = None
    gap_rows: int = 0
    @property
    def id(self):
        return hashlib.sha256(self.train.tobytes()+self.valid.tobytes()+self.kind.encode()).hexdigest()[:16]
    def json(self):
        result={'id':self.id,'kind':self.kind,'reason':self.reason,'seed':self.seed,'train_rows':len(self.train),'valid_rows':len(self.valid)}
        if self.time_column is not None:result.update(time_column=self.time_column,gap_rows=self.gap_rows)
        return result


def _time_window(dates, lower, upper, seed, column, gap, reason):
    idx=np.arange(len(dates))
    a=idx[(dates<lower).to_numpy()]
    mask=dates>=lower
    if upper is not None:mask&=dates<upper
    b=idx[mask.to_numpy()]
    if gap:
        if gap>=len(a)-1:return None
        a=a[np.argsort(dates.iloc[a].to_numpy(),kind='stable')][:-gap]
    if len(a)>1 and len(b):return Split(a,b,'time',reason,seed,column,gap)
    return None


def confirmation_splits(data: Dataset, primary: Split, seed=42, max_windows=2):
    """Freeze score-independent controls; temporal validation windows never overlap."""
    if max_windows<1 or not len(primary.valid):return []
    if primary.kind!='time':
        second=design(data,seed+1009,fraction=.25)
        return [second] if len(second.valid) and second.id!=primary.id else []
    column=primary.time_column
    if column is None:return []
    dates=pd.to_datetime(data.train[column],errors='coerce',utc=True)
    ordered=dates.dropna().sort_values(kind='stable')
    boundary=dates.iloc[primary.valid].min()
    stop=int((ordered<boundary).sum())
    width=max(1,len(primary.valid))
    windows=[]
    upper=boundary
    for number in range(1,max_windows+1):
        position=stop-number*width
        if position<2:break
        lower=ordered.iloc[position]
        if lower>=upper:continue  # A repeated timestamp cannot be split across windows.
        split=_time_window(dates,lower,upper,seed+1009+number,column,primary.gap_rows,
                           f'Expanding-window control {number} in {column}; disjoint from later validation windows')
        if split is None:break
        windows.append(split)
        upper=lower
    return windows


def design(data: Dataset,seed=42,fraction=.2) -> Split:
    train=data.train;spec=data.spec;n=len(train);idx=np.arange(n)
    if n<4: return Split(idx,idx[:0],'unsupported_tiny','Too few rows for honest validation',seed)
    desc=data.inputs.description.lower()
    desc=re.sub(r'[^\n.!?]*(?:not (?:a )?forecast|не прогноз|не является[^\n.!?]*прогноз)[^\n.!?]*[.!?]?', '', desc)
    explicit_time=bool(re.search(r'forecast|forward validation|time.series|прогноз.*будущ|временн.*ряд|будущ.*период|validation\s*:\s*time',desc))
    for col in spec.datetime_candidates:
        dates=pd.to_datetime(train[col],errors='coerce',utc=True)
        if explicit_time and dates.notna().mean()>.95:
            # Split at a timestamp boundary; no equal timestamps straddle the boundary.
            ordered=dates.dropna().sort_values(kind='stable')
            cutoff=ordered.iloc[min(len(ordered)-1,max(1,int(len(ordered)*(1-fraction))))]
            gap_match=re.search(r'gap_rows\s*[:=]\s*(\d+)',desc)
            gap=int(gap_match.group(1)) if gap_match else 0
            split=_time_window(dates,cutoff,None,seed,col,gap,f'Explicit forecasting intent; forward boundary in {col}')
            if split is not None:return split
            if gap:return Split(idx,idx[:0],'unsupported_gap','Required temporal gap leaves too few training rows',seed)
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
