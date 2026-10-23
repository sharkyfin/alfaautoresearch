"""Smoothed target statistics with row/group cross-fitting inside a train fold."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GroupKFold,StratifiedKFold


class CrossfitTargetEncoder:
    def __init__(self,n_classes=0,smoothing=20.,seed=42):
        self.n_classes=n_classes;self.smoothing=smoothing;self.seed=seed
    def _fit(self,x,y):
        target=np.eye(self.n_classes,dtype=np.float64)[np.asarray(y,dtype=int)] if self.n_classes else np.asarray(y,float).reshape(-1,1)
        prior=target.mean(axis=0);maps={}
        for column in x:
            frame=pd.DataFrame(target,columns=range(target.shape[1]));frame['category']=x[column].to_numpy()
            stats=frame.groupby('category',sort=False).agg(['sum','count'])
            values=np.column_stack([(stats[(i,'sum')]+self.smoothing*prior[i])/(stats[(i,'count')]+self.smoothing) for i in range(target.shape[1])])
            maps[column]={key:value for key,value in zip(stats.index,values)}
        return prior,maps
    @staticmethod
    def _transform(x,state):
        prior,maps=state;blocks=[]
        for column in x:
            blocks.append(np.asarray([maps[column].get(value,prior) for value in x[column]],dtype=np.float32))
        return np.hstack(blocks)
    def fit_transform(self,x,y,groups=None):
        x=x.fillna('__MISSING__').astype(str).reset_index(drop=True);y=np.asarray(y)
        count=min(5,len(x))
        if count<2:raise ValueError('Target encoding needs at least two training rows')
        if groups is not None:
            count=min(count,len(np.unique(groups)))
            if count<2:raise ValueError('Target encoding needs independent training groups')
            folds=GroupKFold(count).split(x,y,groups)
        elif self.n_classes and np.unique(y,return_counts=True)[1].min()>=2:
            count=min(count,int(np.unique(y,return_counts=True)[1].min()))
            folds=StratifiedKFold(count,shuffle=True,random_state=self.seed).split(x,y)
        else:folds=KFold(count,shuffle=True,random_state=self.seed).split(x)
        out=np.empty((len(x),x.shape[1]*max(self.n_classes,1)),dtype=np.float32)
        self.fold_records=[]
        for train,valid in folds:
            if groups is not None and set(np.asarray(groups)[train])&set(np.asarray(groups)[valid]):
                raise ValueError('Target-encoding group overlap')
            state=self._fit(x.iloc[train],y[train]);out[valid]=self._transform(x.iloc[valid],state)
            self.fold_records.append({'training_rows':len(train),'validation_rows':len(valid)})
        self.state=self._fit(x,y);self.columns=list(x)
        return out
    def transform(self,x):
        return self._transform(x[self.columns].fillna('__MISSING__').astype(str),self.state)
