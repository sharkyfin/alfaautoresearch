"""Version-compatible CPU wrappers. All models train from scratch."""
from __future__ import annotations
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from .features import FeatureBuilder,MatrixBuilder

FAMILIES={'linear','hist','extra_trees','catboost','lightgbm','xgboost'}


def fit_predict(data,recipe,train_idx,valid_idx,seed=42,threads=8,refit=False,iterations_override=None,feature_function=None,progress_path=None,evaluation_only=False):
    spec=data.spec;cl=spec.classification;family=recipe['model_family'];params=dict(recipe.get('params',{}));ops=recipe.get('feature_ops',[])
    bag=params.pop('seed_bag',1)
    if bag>1:
        import copy
        single=copy.deepcopy(recipe);single['params'].pop('seed_bag')
        fits=[fit_predict(data,single,train_idx,valid_idx,seed+1009*number,threads,refit,
            iterations_override[number] if isinstance(iterations_override,list) else iterations_override,feature_function,progress_path,evaluation_only) for number in range(bag)]
        return (np.mean([fit[0] for fit in fits],axis=0),np.mean([fit[1] for fit in fits],axis=0),
            {'iterations':int(np.median([fit[2]['iterations'] for fit in fits])),
             'bag_iterations':[fit[2]['iterations'] for fit in fits],'features':fits[0][2]['features'],'diagnostics':{'seed_bagging':{
                'seeds':[seed+1009*number for number in range(bag)],'members':bag,
                'fold_train_rows':len(train_idx),'fold_valid_rows':len(valid_idx)}}})
    transform=recipe.get('target_transform','none')
    a=data.train.iloc[train_idx];b=data.train.iloc[valid_idx]
    f=FeatureBuilder(spec,ops,feature_function,recipe.get('feature_expressions',[]))
    za=f.fit_transform(a);zb=f.transform(b);zt=f.transform(data.test)
    encoding=params.pop('category_encoding','native');smoothing=params.pop('target_smoothing',20.)
    encoding_info=None;categoricals=list(f.cat);numericals=list(f.num)
    if encoding=='target':
        from .encoding import CrossfitTargetEncoder
        from pandas.util import hash_pandas_object
        if spec.validation_plan.get('kind')=='time':raise ValueError('Temporal target encoding is unavailable')
        selected=[column for column in f.cat if column not in spec.text_candidates][:12]
        if not selected:raise ValueError('Target encoding requires ordinary categorical features')
        if len(spec.labels)>32 or len(train_idx)>200000:raise ValueError('Target-encoding memory bound')
        groups=None;validation=spec.validation_plan
        if validation.get('kind')=='group':
            groups=a[validation['group_column']].fillna('__MISSING__').astype(str).to_numpy()
        elif validation.get('kind')=='duplicate_group':groups=hash_pandas_object(a[spec.features],index=False).to_numpy()
        target=data.y[train_idx]
        if recipe.get('target_transform')=='log1p':target=np.log1p(target)
        encoder=CrossfitTargetEncoder(len(spec.labels) if cl else 0,smoothing,seed)
        training=encoder.fit_transform(za[selected],target,groups)
        held=encoder.transform(zb[selected]) if len(zb) else np.empty((0,training.shape[1]),dtype=np.float32)
        inference=encoder.transform(zt[selected])
        names=['__atlas_te_'+str(i) for i in range(training.shape[1])]
        for frame,values in [(za,training),(zb,held),(zt,inference)]:
            if set(names)&set(frame):raise ValueError('Target-encoding feature collision')
            frame.drop(columns=selected,inplace=True)
            for i,name in enumerate(names):frame[name]=values[:,i]
        categoricals=[column for column in f.cat if column not in selected];numericals=list(f.num)+names
        encoding_info={'columns':selected,'smoothing':smoothing,'folds':encoder.fold_records,
                       'group_preserving':groups is not None,'training_encoding':'cross-fitted; own row/group targets excluded'}
    text_components=params.pop('text_svd_components',64)
    text_analyzer=params.pop('text_analyzer','word')
    matrix=MatrixBuilder('linear' if family=='linear' else family,categoricals,numericals,spec.text_candidates,text_components,seed,text_analyzer).fit(za)
    xa=matrix.transform(za);xv=matrix.transform(zb) if len(valid_idx) else None;xt=matrix.transform(zt)
    encoded_names=list(xa.columns) if family!='linear' else []
    if family!='linear':
        safe_names=[f'f{i:05d}' for i in range(xa.shape[1])]
        for frame in (xa,xv,xt):
            if frame is not None:frame.columns=safe_names
    y=data.y[train_idx];yv=None if evaluation_only else data.y[valid_idx]
    if transform=='log1p':
        if cl or (y<0).any():raise ValueError('Invalid log target transform')
        y=np.log1p(y)
        if yv is not None:yv=np.log1p(yv)
    weight=params.pop('weight','none');stopping_metric=params.pop('stopping_metric','loss')
    sample_weight=None
    if cl and weight!='none':
        sample_weight=compute_sample_weight('balanced',y)
        if weight=='sqrt':sample_weight=np.sqrt(sample_weight)
    rounds=iterations_override or params.pop('iterations',500)
    if iterations_override is not None:params.pop('iterations',None)
    loss='absolute_error' if spec.metric.name=='mae' else 'squared_error'
    common=dict(random_state=seed)
    fit_args={}
    if family=='catboost':
        from catboost import CatBoostClassifier,CatBoostRegressor
        objective=('MultiClass' if len(spec.labels)>2 else 'Logloss') if cl else 'MAE' if spec.metric.name=='mae' else 'RMSE'
        model=(CatBoostClassifier if cl else CatBoostRegressor)(
            iterations=rounds,depth=params.pop('depth',6),learning_rate=params.pop('learning_rate',.05),
            loss_function=objective,task_type='CPU',thread_count=threads,random_seed=seed,
            allow_writing_files=False,verbose=False,**params)
        fit_args['cat_features']=[i for i,name in enumerate(encoded_names) if name in f.cat]
        if len(valid_idx) and not refit and not evaluation_only:
            fit_args.update(eval_set=(xv,yv),early_stopping_rounds=60,use_best_model=True)
    elif family=='lightgbm':
        import lightgbm as lgb
        objective=('multiclass' if len(spec.labels)>2 else 'binary') if cl else 'regression_l1' if spec.metric.name=='mae' else 'regression'
        model=(lgb.LGBMClassifier if cl else lgb.LGBMRegressor)(n_estimators=rounds,learning_rate=params.pop('learning_rate',.045),num_leaves=params.pop('num_leaves',31),verbosity=-1,n_jobs=threads,objective=objective,deterministic=True,force_col_wise=True,**common,**params)
        if params.get('subsample',1.)<1.:model.set_params(subsample_freq=1)
        if len(valid_idx) and not refit and not evaluation_only:
            if stopping_metric=='primary' and cl:
                classes=np.unique(y)
                def primary_metric(y_true,probabilities):
                    if probabilities.ndim==1:probabilities=np.column_stack([1-probabilities,probabilities])
                    aligned=np.zeros((len(probabilities),len(spec.labels)))
                    aligned[:,classes]=probabilities
                    actual_y=classes[np.asarray(y_true,dtype=int)]
                    return 'primary_'+spec.metric.name,spec.metric.score(actual_y,aligned),spec.metric.direction>0
                model.set_params(metric='None')
                fit_args['eval_metric']=primary_metric
            fit_args.update(eval_set=[(xv,yv)],callbacks=[lgb.early_stopping(100 if stopping_metric=='primary' else 60,first_metric_only=True,verbose=False),lgb.log_evaluation(0)])
    elif family=='xgboost':
        from xgboost import XGBClassifier,XGBRegressor
        objective='multi:softprob' if cl and len(spec.labels)>2 else 'binary:logistic' if cl else 'reg:absoluteerror' if spec.metric.name=='mae' else 'reg:squarederror'
        model=(XGBClassifier if cl else XGBRegressor)(n_estimators=rounds,learning_rate=params.pop('learning_rate',.05),max_depth=params.pop('max_depth',6),tree_method='hist',n_jobs=threads,objective=objective,**common,**params)
        if len(valid_idx) and not refit and not evaluation_only:model.set_params(early_stopping_rounds=60);fit_args['eval_set']=[(xv,yv)];fit_args['verbose']=False
    elif family=='hist':
        from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
        model=(HistGradientBoostingClassifier if cl else HistGradientBoostingRegressor)(max_iter=rounds,learning_rate=params.pop('learning_rate',.08),max_leaf_nodes=params.pop('max_leaf_nodes',31),early_stopping=False,**common,**params,**({} if cl else {'loss':loss}))
    elif family=='extra_trees':
        from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
        model=(ExtraTreesClassifier if cl else ExtraTreesRegressor)(n_estimators=rounds,min_samples_leaf=params.pop('min_samples_leaf',2),max_features=params.pop('max_features',.8),n_jobs=threads,**common,**params)
    elif family=='linear':
        if cl:
            from sklearn.linear_model import LogisticRegression
            model=LogisticRegression(C=params.pop('C',1),max_iter=400,solver='lbfgs',**params)
        else:
            from sklearn.linear_model import Ridge
            model=Ridge(alpha=params.pop('alpha',10),solver='lsqr',**params)
    else:raise ValueError('Unsupported model '+family)
    if sample_weight is not None:fit_args['sample_weight']=sample_weight
    if progress_path is not None:
        import json
        from pathlib import Path
        Path(progress_path).write_text(json.dumps({'state':'fitting','model_family':family,'rows':len(y)}))
    model.fit(xa,y,**fit_args)
    def predict(x):
        if not cl:
            p=model.predict(x)
            if transform=='log1p':p=np.expm1(np.clip(p,-30,50))
            return np.maximum(p,0) if spec.metric.name=='rmsle' else p
        p=model.predict_proba(x);out=np.zeros((x.shape[0],len(spec.labels)))
        for i,c in enumerate(model.classes_):out[:,int(c)]=p[:,i]
        out=np.maximum(out,1e-12);return out/out.sum(1,keepdims=True)
    pv=predict(xv) if len(valid_idx) else np.empty((0,len(spec.labels))) if cl else np.empty(0)
    pt=predict(xt)
    best=getattr(model,'best_iteration_',None) or getattr(model,'tree_count_',None) or rounds
    if family=='catboost':
        best_iteration=model.get_best_iteration()
        best=best_iteration+1 if best_iteration is not None and best_iteration>=0 else model.tree_count_
    if family=='linear':
        count=getattr(model,'n_iter_',None)
        if count is not None:best=int(np.max(count))
    if hasattr(model,'best_iteration'):
        try:best=model.best_iteration+1
        except AttributeError:pass
    importance=getattr(model,'feature_importances_',None)
    diagnostics={'feature_importance':dict(sorted(zip(encoded_names,map(float,importance)),key=lambda t:-t[1])[:15])} if importance is not None else {}
    if encoding_info:diagnostics['target_encoding']=encoding_info
    text_info={c:{'components':p.components,'vocabulary':len(p.vectorizer.vocabulary_),'fit_rows':p.fit_rows} for c,p in matrix.text_projections.items()}
    if text_info:diagnostics['text_projections']=text_info
    train_metric=None
    try:
        sample=xa.iloc[:2048] if hasattr(xa,'iloc') else xa[:2048]
        train_metric=spec.metric.score(data.y[train_idx][:2048],predict(sample))
    except ValueError as error:
        diagnostics['training_metric_unavailable']=str(error)[:200]
    return pv,pt,{'iterations':int(best),'features':xa.shape[1],'diagnostics':diagnostics,'train_sample_metric':train_metric,'fit_validation_rows':0 if evaluation_only or refit else len(valid_idx),
        'fit_role':'refit' if refit else 'confirmation' if evaluation_only else 'search'}
