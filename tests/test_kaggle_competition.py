"""Original competition identities and hidden-target separation must survive export."""
import pandas as pd
import pytest

from scripts.kaggle_competition import RAW_FILES, prepare_competition, validate_competition_output


@pytest.fixture
def competition(tmp_path):
    raw=tmp_path/'raw'; raw.mkdir()
    train=pd.DataFrame({'PassengerId':[f'{i:04}_01' for i in range(1,7)],
                        'HomePlanet':['Earth','Mars',None,'Earth','Mars','Earth'],
                        'Age':[20.,35.,40.,22.,18.,50.],
                        'Transported':[True,False,True,False,True,False]})
    test=pd.DataFrame({'PassengerId':['0009_01','0010_01'],'HomePlanet':['Earth',None],'Age':[21.,31.]})
    sample=pd.DataFrame({'PassengerId':test.PassengerId,'Transported':[False,False]})
    for name,frame in zip(RAW_FILES,[train,test,sample]):frame.to_csv(raw/name,index=False)
    spec={'slug':'fixture','source':'https://www.kaggle.com/competitions/fixture','metric':'accuracy',
          'target':'Transported','id_column':'PassengerId','labels':['False','True'],
          'training_rows':len(train),'test_rows':len(test),'description':'Predict the Boolean target.',
          'columns':{column:column for column in train}}
    return raw,spec


def test_competition_preparation_preserves_original_files_and_private_test(competition,tmp_path):
    raw,spec=competition
    task=tmp_path/'task'
    metadata=prepare_competition(raw,task,spec)
    assert all((task/name).read_bytes()==(raw/name).read_bytes() for name in RAW_FILES)
    assert 'Transported' not in pd.read_csv(task/'test.csv')
    assert len(pd.read_csv(task/'train.csv'))==6 and len(pd.read_csv(task/'test.csv'))==2
    dictionary=pd.read_csv(task/'data_dictionary.csv').set_index('feature')
    assert dictionary.loc['PassengerId','role']=='id' and dictionary.loc['Transported','role']=='target'
    assert metadata['original_split_preserved'] and not metadata['test_labels_available']
    output=tmp_path/'prediction.csv'
    pd.DataFrame({'PassengerId':['0009_01','0010_01'],'Transported':[True,False]}).to_csv(output,index=False)
    result=validate_competition_output(output,task/'sample_submission.csv',spec)
    assert result['rows']==2 and result['label_counts']=={'True':1,'False':1}


@pytest.mark.parametrize('corruption',['test_target','overlapping_ids','sample_order','missing_label'])
def test_competition_preparation_rejects_leaks_or_changed_identities(competition,tmp_path,corruption):
    raw,spec=competition
    if corruption=='test_target':
        frame=pd.read_csv(raw/'test.csv');frame['Transported']=[True,False];frame.to_csv(raw/'test.csv',index=False)
    elif corruption=='overlapping_ids':
        frame=pd.read_csv(raw/'test.csv');frame.loc[0,'PassengerId']='0001_01';frame.to_csv(raw/'test.csv',index=False)
    elif corruption=='sample_order':
        pd.read_csv(raw/'sample_submission.csv').iloc[::-1].to_csv(raw/'sample_submission.csv',index=False)
    else:
        frame=pd.read_csv(raw/'train.csv');frame['Transported']=frame.Transported.astype(object)
        frame.loc[0,'Transported']=None;frame.to_csv(raw/'train.csv',index=False)
    with pytest.raises(ValueError):prepare_competition(raw,tmp_path/'task',spec)


@pytest.mark.parametrize('corruption',['row_order','probabilities','duplicate_id','missing_prediction'])
def test_manual_submission_rejects_bad_ids_and_non_boolean_predictions(competition,tmp_path,corruption):
    raw,spec=competition
    frame=pd.read_csv(raw/'sample_submission.csv',dtype=str)
    if corruption=='row_order':frame=frame.iloc[::-1]
    elif corruption=='probabilities':frame['Transported']=['0.9','0.1']
    elif corruption=='duplicate_id':frame.loc[1,'PassengerId']=frame.loc[0,'PassengerId']
    else:frame.loc[1,'Transported']=None
    output=tmp_path/'bad.csv';frame.to_csv(output,index=False)
    with pytest.raises(ValueError):validate_competition_output(output,raw/'sample_submission.csv',spec)
