"""Immutable, common split plan. Never select a split by its model score."""
from __future__ import annotations
from dataclasses import dataclass,replace
import hashlib,re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit,GroupKFold,KFold,StratifiedKFold,StratifiedGroupKFold,train_test_split
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
    group_column: str | None = None
    class_support: dict | None = None
    @property
    def id(self):
        return hashlib.sha256(self.train.tobytes()+self.valid.tobytes()+self.kind.encode()).hexdigest()[:16]
    def json(self):
        result={'id':self.id,'kind':self.kind,'reason':self.reason,'seed':self.seed,'train_rows':len(self.train),'valid_rows':len(self.valid)}
        if self.time_column is not None:result.update(time_column=self.time_column,gap_rows=self.gap_rows)
        if self.group_column is not None:result['group_column']=self.group_column
        if self.class_support is not None:result['class_support']=self.class_support
        return result


def protected_groups(data,groups):
    """Keep an entire entity/duplicate group in train when it owns a class."""
    protected=[]
    if data.spec.classification:
        for label in np.unique(data.y):
            owners=np.unique(groups[data.y==label])
            if len(owners)==1:protected.extend(owners.tolist())
    return np.asarray(protected,dtype=groups.dtype)


def grouped_holdout(data,groups,fraction,seed,kind,reason,column=None):
    indices=np.arange(len(data.y))
    if len(np.unique(groups))<2:
        return Split(indices,indices[:0],kind,reason+'; no independent group for validation',seed,group_column=column)
    if not data.spec.classification:
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(indices,groups=groups))
        return Split(a,b,kind,reason,seed,group_column=column)
    fixed=protected_groups(data,groups)
    pool=indices[~np.isin(groups,fixed)]
    count=len(np.unique(groups[pool]))
    if count<2:return Split(indices,indices[:0],kind,reason+'; insufficient independent class-bearing groups',seed,group_column=column)
    folds=min(count,max(2,round(1/fraction)))
    supports=np.unique(data.y[pool],return_counts=True)[1]
    if len(supports) and supports.min()>=2:
        folds=min(folds,int(supports.min()))
        splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    else:splitter=GroupKFold(n_splits=folds)
    for _,held in splitter.split(pool,data.y[pool],groups[pool]):
        valid=pool[held];train=np.setdiff1d(indices,valid)
        if set(data.y[train])==set(data.y):
            return Split(train,valid,kind,reason+'; class-aware groups, protected class owners train-only',seed,group_column=column)
    return Split(indices,indices[:0],kind,reason+'; no feasible class-preserving group holdout',seed,group_column=column)


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
    """Freeze controls with disjoint validation rows and preserved groups/time."""
    if max_windows<1 or not len(primary.valid):return []
    if primary.kind!='time':
        indices=np.arange(len(data.y));pool=np.asarray(primary.train)
        groups=None
        if primary.kind=='group':
            if primary.group_column is None:return []
            groups=data.train[primary.group_column].fillna('__MISSING__').astype(str).to_numpy()
        elif primary.kind=='duplicate_group':
            groups=pd.util.hash_pandas_object(data.train[data.spec.features],index=False).to_numpy()
        # Removing only a rare row would split its entity between train/valid.
        if groups is not None:
            pool=pool[~np.isin(groups[pool],protected_groups(data,groups))]
        elif data.spec.classification:
            counts=np.bincount(data.y,minlength=len(data.spec.labels))
            pool=pool[counts[data.y[pool]]>=2]
        if len(pool)<4:return []
        if groups is not None:
            if len(np.unique(groups[pool]))<2:return []
            can_stratify=data.spec.classification and np.unique(data.y[pool],return_counts=True)[1].min()>=2
            splitter=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=seed+1009) if can_stratify else GroupKFold(n_splits=2)
            folds=splitter.split(pool,data.y[pool],groups[pool])
        elif data.spec.classification and np.unique(data.y[pool],return_counts=True)[1].min()>=2:
            folds=StratifiedKFold(n_splits=2,shuffle=True,random_state=seed+1009).split(pool,data.y[pool])
        else:
            folds=KFold(n_splits=2,shuffle=True,random_state=seed+1009).split(pool)
        out=[]
        for number,(_,held) in enumerate(folds,1):
            valid=pool[held];train=np.setdiff1d(indices,valid)
            if data.spec.classification and set(data.y[train])!=set(data.y):continue
            out.append(Split(train,valid,primary.kind,
                'Frozen cross-validation control '+str(number)+'; validation disjoint from search and other controls',
                seed+1009+number,group_column=primary.group_column))
        return out[:max_windows]
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


def selection_splits(data: Dataset,seed=42):
    """One search fold, then two frozen confirmation folds for the finalists."""
    primary=design(data,seed)
    if primary.kind not in {'time','unsupported_tiny','unsupported_gap'}:
        primary=design(data,seed,fraction=1/3)
    controls=confirmation_splits(data,primary,seed)
    if data.spec.classification:
        def support(split):
            fit=np.bincount(data.y[split.train],minlength=len(data.spec.labels))
            held=np.bincount(data.y[split.valid],minlength=len(data.spec.labels))
            return replace(split,class_support={'train':fit.tolist(),'valid':held.tolist(),
                'unvalidated_class_indices':np.flatnonzero(held==0).tolist()})
        primary=support(primary);controls=[support(split) for split in controls]
    return primary,controls


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
        if groups.nunique()>=2 and groups.nunique()<n*.8 and (unseen>.8 or explicit):
            return grouped_holdout(data,groups.to_numpy(),fraction,seed,'group',f'Repeated {col}; new test entity fraction {unseen:.3f}',col)
    # Exact duplicate feature rows cannot leak across random folds.
    if spec.features:
        fingerprints=pd.util.hash_pandas_object(train[spec.features],index=False).to_numpy()
        sizes=pd.Series(fingerprints).value_counts()
        conflicts=pd.DataFrame({'group':fingerprints,'target':data.y}).groupby('group')['target'].nunique()
        spec.profile['duplicate_audit']={'duplicate_row_fraction':float((sizes[sizes>1].sum()/n)),
            'duplicate_groups':int((sizes>1).sum()),'conflicting_target_groups':int((conflicts>1).sum())}
        if len(np.unique(fingerprints))<n:
            return grouped_holdout(data,fingerprints,fraction,seed,'duplicate_group','Exact feature duplicates held together')
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
