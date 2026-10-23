"""Strict schema checks and same-filesystem atomic checkpoints."""
from __future__ import annotations
import os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .task import Dataset,read_csv


def make_submission(data: Dataset, prediction) -> pd.DataFrame:
    spec=data.spec;p=np.asarray(prediction)
    if not np.isfinite(p).all(): raise ValueError('Predictions contain NaN/Inf')
    if len(p)!=len(data.test): raise ValueError('Incorrect prediction row count')
    out=data.test[spec.ids].copy() if spec.ids else pd.DataFrame(index=np.arange(len(data.test)))
    if spec.classification:
        if p.shape!=(len(data.test),len(spec.labels)): raise ValueError('Probability shape mismatch')
        if (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError('Invalid class probabilities')
        if spec.prediction_kind=='probability':
            for c,i in spec.probability_columns.items(): out[c]=p[:,i]
        else: out[spec.target]=np.asarray(spec.labels)[p.argmax(1)]
    else:
        out[spec.target]=p.reshape(-1)
        if spec.metric.name=='rmsle': out[spec.target]=out[spec.target].clip(lower=0)
    if spec.ids:
        # Preserve sample order, even when test input has a different ID order.
        out=data.sample[spec.ids].merge(out,on=spec.ids,how='left',validate='one_to_one',sort=False)
    out=out.loc[:,data.sample.columns]
    validate_frame(data,out)
    return out


def validate_frame(data: Dataset,frame: pd.DataFrame):
    spec=data.spec
    if list(frame.columns)!=list(data.sample.columns) or len(frame)!=len(data.sample): raise ValueError('Output schema mismatch')
    if frame.isna().any().any(): raise ValueError('Missing output value')
    for c in spec.ids:
        if not frame[c].astype(str).reset_index(drop=True).equals(data.sample[c].astype(str).reset_index(drop=True)): raise ValueError('ID order mismatch')
    for c in frame:
        if c in spec.ids: continue
        if spec.classification and spec.prediction_kind=='label':
            if not frame[c].isin(spec.labels).all(): raise ValueError('Invalid label')
        elif not np.isfinite(pd.to_numeric(frame[c],errors='coerce')).all(): raise ValueError('Nonfinite output')
    if spec.classification and spec.prediction_kind=='probability':
        values=frame[list(spec.probability_columns)].to_numpy(dtype=float)
        if (values<0).any() or (values>1+1e-8).any():raise ValueError('Output probability outside [0,1]')
        if values.shape[1]==len(spec.labels) and not np.allclose(values.sum(1),1.,atol=1e-5):
            raise ValueError('Output class probabilities do not sum to one')


def checkpoint(data: Dataset,prediction,path: Path | None=None):
    path=path or data.inputs.output;path.parent.mkdir(parents=True,exist_ok=True)
    frame=make_submission(data,prediction)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            frame.to_csv(f,index=False);f.flush();os.fsync(f.fileno())
        loaded=read_csv(tmp,dtype={c:'string' for c in data.spec.ids})
        validate_frame(data,loaded)
        os.replace(tmp,path)
        if hasattr(os,'O_DIRECTORY'):
            dfd=os.open(path.parent,os.O_DIRECTORY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return path


def dummy_prediction(y,spec,n):
    if spec.classification:
        counts=np.bincount(y.astype(int),minlength=len(spec.labels));return np.tile(counts/counts.sum(),(n,1))
    scale=max(float(np.max(np.abs(y))),1.)
    if spec.metric.name=='mape':
        order=np.argsort(y);weights=1/np.maximum(np.abs(y[order]),np.finfo(float).eps)
        value=y[order[np.searchsorted(np.cumsum(weights/weights.sum()),.5)]]
        return np.full(n,value,dtype=float)
    value=np.median(y/scale)*scale if spec.metric.name=='mae' else np.expm1(np.log1p(y).mean()) if spec.metric.name=='rmsle' else np.mean(y/scale)*scale
    return np.full(n,value,dtype=float)
