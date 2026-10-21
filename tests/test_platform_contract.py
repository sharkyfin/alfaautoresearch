"""Platform paths and reaching AutoResearch within the fast portfolio."""
import shutil

import pytest

from atlas.budget import Budget
from atlas.contract import read_inputs
from atlas.search import SearchController
from atlas.task import load
from scripts.synthetic import create


@pytest.mark.parametrize('syntax', ['code', 'markdown', 'plain'])
def test_nested_input_paths_accept_markdown_and_plain_text(tmp_path, syntax):
    task=create(tmp_path)
    directory=tmp_path/'participants';directory.mkdir()
    paths=[]
    for filename in ['train.csv','test.csv','data_dictionary.csv','sample_submission.csv']:
        shutil.move(tmp_path/filename,directory/filename)
        path='participants/'+filename
        paths.append('`'+path+'`' if syntax=='code' else '[data]('+path+')' if syntax=='markdown' else path)
    task.write_text('Task: regression\nPrimary metric: RMSE\n'+'\n'.join(paths)+'\nWrite submission.csv.\n')
    inputs=read_inputs(task)
    assert inputs.train==directory/'train.csv'
    assert inputs.sample==directory/'sample_submission.csv'
    assert inputs.output==tmp_path/'submission.csv'
    assert len(load(inputs).train)>0


def test_platform_output_cannot_be_redirected_by_task_prose(tmp_path):
    task=create(tmp_path)
    task.write_text(task.read_text()+'\nHistorical output examples: `reports/predictions.csv` and `answer.csv`.\n')
    assert read_inputs(task).output==tmp_path/'submission.csv'


def test_missing_described_paths_fall_back_to_platform_root(tmp_path):
    task=create(tmp_path)
    task.write_text(task.read_text()+'\nOld example: `unavailable/train.csv`.\n')
    assert read_inputs(task).train==tmp_path/'train.csv'


def test_csv_download_urls_do_not_become_local_absolute_paths(tmp_path):
    task=create(tmp_path)
    task.write_text(task.read_text()+'\nReference https://example.invalid/data/train.csv and [reference](https://example.invalid/test.csv).\n')
    assert read_inputs(task).train==tmp_path/'train.csv'


def test_inline_code_paths_with_spaces_are_not_parsed_twice(tmp_path):
    task=create(tmp_path)
    directory=tmp_path/'input files';directory.mkdir()
    shutil.move(tmp_path/'train.csv',directory/'train.csv')
    task.write_text(task.read_text().replace('`train.csv`','`input files/train.csv`'))
    assert read_inputs(task).train==directory/'train.csv'


def test_long_near_matching_plain_path_does_not_expand_regex_backtracking(tmp_path):
    task=create(tmp_path)
    task.write_text(task.read_text()+'\nExample: '+('segment/'*1000)+'train.csv_old\n')
    assert read_inputs(task).train==tmp_path/'train.csv'


def test_ambiguous_or_outside_inputs_are_not_silently_selected(tmp_path):
    task=create(tmp_path)
    directory=tmp_path/'second';directory.mkdir()
    shutil.copy2(tmp_path/'train.csv',directory/'train.csv')
    task.write_text(task.read_text()+'\nAnother `second/train.csv`.\n')
    with pytest.raises(ValueError,match='Ambiguous'):
        read_inputs(task)
    task.write_text('Inputs: `../train.csv`.\n')
    with pytest.raises(ValueError,match='outside'):
        read_inputs(task)


def test_live_research_runs_after_two_baselines_before_rest_of_portfolio(tmp_path):
    data=load(read_inputs(create(tmp_path)))
    scheduler=SearchController(data.spec,Budget(900),mode='live-agent')
    first=scheduler.next(None)
    champion={'experiment_id':'first','recipe':first}
    second=scheduler.next(champion)
    assert scheduler.stage=='MODEL_DIVERSITY'
    remaining=len(scheduler.queue)
    hypothesis=scheduler.next(champion)
    assert scheduler.stage=='RESEARCH' and hypothesis['parent_experiment']=='first'
    assert len(scheduler.queue)==remaining>0
    scheduler.next(champion)
    assert scheduler.stage=='MODEL_DIVERSITY'


def test_old_research_feedback_survives_baselines_and_context_compaction(tmp_path):
    from atlas.agent import compact_context,researcher_context
    data=load(read_inputs(create(tmp_path)))
    experiment={'experiment_id':'research','origin':'qwen','status':'ok','metric_value':1.,
                'recipe':{'model_family':'hist'},'diagnostics':{'residual_bias':.1}}
    history=[experiment]+[{'experiment_id':'baseline_'+str(i),'origin':'deterministic','status':'ok',
                          'recipe':{'model_family':'linear'},'diagnostics':{'feature_importance':{'long_column_'+str(j):1. for j in range(80)}}} for i in range(10)]
    context=researcher_context(data,history,None,100)
    assert any(r['experiment_id']=='research' for r in context['history'])
    context['untrusted_data']['dictionary']=[{'column':'x'+str(i),'description':'Long column description '*30} for i in range(60)]
    compact=compact_context(context,limit=6500)
    assert any(r['experiment_id']=='research' and r['metric_value']==1. for r in compact['history'])
