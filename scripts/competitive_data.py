"""Development evaluator adapters: UCI APS and forward Bike; no agent imports."""
from __future__ import annotations
from pathlib import Path
import hashlib,io,json,zipfile
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path('/Users/mac/Documents/projects/alfaautoresearch')
FILES=['train.csv','test.csv','data_dictionary.csv','sample_submission.csv','task.md']

def prepare(case,destination,phase='development',seed=42):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    if case=='aps_failure':
        archive=ROOT/'artifacts/data/final_competitive/aps_failure.zip'
        with zipfile.ZipFile(archive) as z:
            name=next(n for n in z.namelist() if n.endswith('aps_failure_training_set.csv'))
            raw=z.read(name).decode();start=raw.index('class,')
            source=pd.read_csv(io.StringIO(raw[start:]),na_values=['na'])
            idx=np.arange(len(source));fit,held=train_test_split(idx,test_size=.2,stratify=source['class'],random_state=seed)
            if phase=='final':
                fit=idx
                name=next(n for n in z.namelist() if n.endswith('aps_failure_test_set.csv'))
                raw=z.read(name).decode();final=pd.read_csv(io.StringIO(raw[raw.index('class,'):]),na_values=['na'])
                training=source.copy();evaluation=final.copy()
            else:training=source.iloc[fit].copy();evaluation=source.iloc[held].copy()
        target='class';labels=['neg','pos'];metric='average_precision';regime='iid'
        training=training.rename(columns={target:'target'});evaluation=evaluation.rename(columns={target:'target'});target='target'
        training.insert(0,'id',[f'train_{i}' for i in range(len(training))]);evaluation.insert(0,'id',[f'eval_{i}' for i in range(len(evaluation))])
        description='Binary classification of APS component failures in trucks. Primary metric: average_precision. Positive class: pos\nPredict probability of pos in target. Negative label neg. Numeric sensor counters and histogram bins; missing sensor readings are empty. Validation: IID.'
        cat=[]
    elif case=='bike_sharing_temporal':
        archive=ROOT/'artifacts/data/final_competitive/bike_sharing.zip'
        with zipfile.ZipFile(archive) as z:
            source=pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith('hour.csv'))))
        source['timestamp']=pd.to_datetime(source['dteday'])+pd.to_timedelta(source['hr'],unit='h')
        source=source.sort_values('timestamp',kind='stable').reset_index(drop=True)
        # casual + registered equals cnt. Neither is available at prediction time.
        source=source.drop(columns=['casual','registered','instant','dteday']).rename(columns={'cnt':'target'})
        source['timestamp']=source['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        last=int(len(source)*.8);middle=int(len(source)*.64)
        if phase=='final':training=source.iloc[:last].copy();evaluation=source.iloc[last:].copy()
        else:training=source.iloc[:middle].copy();evaluation=source.iloc[middle:last].copy()
        training.insert(0,'id',[f'train_{i}' for i in range(len(training))]);evaluation.insert(0,'id',[f'eval_{i}' for i in range(len(evaluation))])
        target='target';labels=[];metric='rmsle';regime='temporal'
        description='Forecast hourly bike rental counts into future dates using timestamp, calendar and weather. Primary metric: RMSLE. Validation: time. Target is a nonnegative count. casual and registered are unavailable future target components and have been removed. Preserve chronological order; no future-label features.'
        cat=['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']
    else:raise ValueError(case)
    training=training.reset_index(drop=True);evaluation=evaluation.reset_index(drop=True)
    truth=evaluation[['id',target]].copy();features=[c for c in training if c not in ['id',target]]
    training.to_csv(destination/'train.csv',index=False);evaluation.drop(columns=target).to_csv(destination/'test.csv',index=False)
    pd.DataFrame({'id':evaluation['id'],target:0.}).to_csv(destination/'sample_submission.csv',index=False)
    descriptions={'timestamp':'Chronological observation timestamp','hr':'Hour of day (0-23)','season':'Season code','mnth':'Calendar month','yr':'Year indicator','weekday':'Day of week','workingday':'Working-day indicator','holiday':'Holiday indicator','weathersit':'Weather situation code','temp':'Normalized air temperature','atemp':'Normalized apparent temperature','hum':'Relative humidity','windspeed':'Normalized wind speed'}
    rows=[]
    for column in training:
        rows.append({'feature':column,'role':'id' if column=='id' else 'target' if column==target else 'feature',
                     'dtype':'categorical' if column in cat else 'datetime' if column=='timestamp' else 'numeric',
                     'description':descriptions.get(column,'Anonymized sensor counter/bin' if case=='aps_failure' else column),
                     'validation_role':'time' if column=='timestamp' else ''})
    pd.DataFrame(rows).to_csv(destination/'data_dictionary.csv',index=False)
    (destination/'task.md').write_text(description+'\n\nFiles: `train.csv`, `test.csv`, `data_dictionary.csv`, `sample_submission.csv`.\nWrite `submission.csv` with the same IDs and header as sample_submission.csv.\n')
    hashes={n:hashlib.sha256((destination/n).read_bytes()).hexdigest() for n in FILES}
    rawsha=hashlib.sha256(archive.read_bytes()).hexdigest()
    meta={'case':case,'source':'uci','dataset':case,'license':'CC BY 4.0','metric':metric,'original_source_metric':'asymmetric_misclassification_cost' if case=='aps_failure' else 'count_regression_no_fixed_metric',
          'metric_override':'Study uses AP for APS (original challenge used asymmetric cost); RMSLE for Bike counts',
          'target':target,'id_column':'id','labels':labels,'n_classes':len(labels),'seed':seed,'phase':phase,
          'training_rows':len(training),'evaluation_rows':len(evaluation),'features':len(features),'input_sha256':hashes,
          'raw_sha256':rawsha,'split_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
          'split_regime':regime,'labels_storage':'Independent evaluator memory only; no label artifact exposed to child',
          'raw_source':'https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks' if case=='aps_failure' else 'https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset',
          'development_only':phase=='development','final_scope':'APS official held-out 16000 rows; Bike untouched final chronological 20%'}
    return meta,truth
