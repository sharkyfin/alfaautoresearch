"""Trusted metric implementation; candidates cannot choose or edit the objective."""
from __future__ import annotations
from dataclasses import dataclass, field
import re
import numpy as np
from sklearn import metrics as sk

REGRESSION = {'rmse','mse','mae','r2','rmsle'}
CLASSIFICATION = {'accuracy','balanced_accuracy','f1','f1_macro','f1_micro','f1_weighted','roc_auc','logloss','average_precision'}
PROBABILITY = {'roc_auc','logloss','average_precision'}

class UnsupportedTask(ValueError):
    """Task contract does not describe a supported unambiguous prediction problem."""

@dataclass(frozen=True)
class Metric:
    name: str
    n_classes: int = 0
    positive_index: int = 1
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in REGRESSION | CLASSIFICATION: raise UnsupportedTask(f'Unknown metric: {self.name}')
        if self.name=='f1' and self.n_classes!=2: raise UnsupportedTask('Binary F1 requires two classes')
        if self.name=='average_precision' and self.n_classes!=2: raise UnsupportedTask('Average precision: binary only')

    @property
    def direction(self): return -1 if self.name in {'rmse','mse','mae','rmsle','logloss'} else 1
    @property
    def needs_proba(self): return self.name in PROBABILITY
    def utility(self, value): return self.direction * float(value)
    def score(self, y, prediction):
        y=np.asarray(y);p=np.asarray(prediction)
        if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
        if self.n_classes:
            if p.ndim!=2 or p.shape!=(len(y),self.n_classes): raise ValueError('Probability class alignment/shape invalid')
            if (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError('Invalid probabilities')
            hard=p.argmax(1);labels=np.arange(self.n_classes)
            if self.name=='accuracy': value=sk.accuracy_score(y,hard)
            elif self.name=='balanced_accuracy': value=sk.balanced_accuracy_score(y,hard)
            elif self.name.startswith('f1'):
                avg=self.name.removeprefix('f1_') if '_' in self.name else 'binary'
                value=sk.f1_score(y,hard,average=avg,labels=labels,zero_division=0,pos_label=self.positive_index)
            elif self.name=='logloss': value=sk.log_loss(y,np.clip(p,1e-15,1),labels=labels)
            elif self.name=='roc_auc':
                if len(np.unique(y))!=self.n_classes: raise ValueError('AUC undefined: class missing in validation')
                value=sk.roc_auc_score((y==self.positive_index).astype(int),p[:,self.positive_index]) if self.n_classes==2 else sk.roc_auc_score(y,p,labels=labels,multi_class=self.kwargs.get('multi_class','ovr'),average=self.kwargs.get('average','macro'))
            elif self.name=='average_precision':
                if len(np.unique(y))<2: raise ValueError('AP undefined: one class')
                value=sk.average_precision_score(y==self.positive_index,p[:,self.positive_index])
            else: raise UnsupportedTask(f'Classification metric {self.name}')
        else:
            p=p.reshape(-1)
            if len(p)!=len(y): raise ValueError('Prediction size mismatch')
            if self.name=='rmse': value=np.sqrt(sk.mean_squared_error(y,p))
            elif self.name=='mse': value=sk.mean_squared_error(y,p)
            elif self.name=='mae': value=sk.mean_absolute_error(y,p)
            elif self.name=='r2': value=sk.r2_score(y,p)
            elif self.name=='rmsle':
                if (y<0).any() or (p<0).any(): raise ValueError('RMSLE requires nonnegative values')
                value=np.sqrt(sk.mean_squared_log_error(y,p))
            else: raise UnsupportedTask(f'Regression metric {self.name}')
        if not np.isfinite(value): raise ValueError('Metric undefined')
        return float(value)

def infer_metric(description: str) -> tuple[str | None,dict]:
    # Look at primary-metric sentences first; auxiliary metrics do not override them.
    text=description.lower().replace('ё','е')
    patterns=[('rmsle',r'\brmsle\b'),('rmse',r'\brmse\b'),('mae',r'\bmae\b'),('mse',r'\bmse\b'),('r2',r'\br2\b|r²'),('f1_macro',r'macro[ _-]*f1|f1[ _-]*macro|average\s*=\s*["\x27]macro|макро.?f1'),('f1_weighted',r'weighted[ _-]*f1|f1[ _-]*weighted'),('f1_micro',r'micro[ _-]*f1|f1[ _-]*micro'),('logloss',r'log[ _-]*loss|cross[ _-]*entropy'),('roc_auc',r'roc[ _-]*auc|\bauc\b'),('average_precision',r'average[ _-]*precision|\bpr[ _-]*auc\b'),('balanced_accuracy',r'balanced[ _-]*accuracy'),('accuracy',r'\baccuracy\b'),('f1',r'\bf1\b')]
    lines=text.splitlines()
    primary=[l for l in lines if re.search(r'основн.*метрик|primary.*metric|^metric\s*:|^метрика\s*:',l)]
    sections=primary if primary else [text]
    for section in sections:
        for name,pattern in patterns:
            if re.search(pattern,section):
                kwargs={}
                if name=='roc_auc':
                    if 'ovo' in section: kwargs['multi_class']='ovo'
                    if 'weighted' in section: kwargs['average']='weighted'
                return name,kwargs
    return None,{}
