"""Report the four frozen observations without launching or imputing ML runs."""
from pathlib import Path
import argparse,json,hashlib

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reports',type=Path,required=True)
    args=parser.parse_args();out=args.reports
    paired=json.loads((out/'paired-quality.json').read_text());state=json.loads((out/'state.json').read_text())
    if paired['status']!='completed' or len(paired['runs'])!=4 or state['new_stage_ml_runs_used']!=4:
        raise ValueError('Require four completed observations; do not invent missing cells')
    details=[]
    for row in paired['runs']:
        run=Path(row['run_dir']);summary=json.loads((run/'summary.json').read_text())
        experiments=[json.loads(x) for x in (run/'experiments.jsonl').read_text().splitlines()]
        ag=[r for r in experiments if r['recipe']['model_family']=='autogluon']
        details.append({'case':row['case'],'arm':row['arm'],'external_metric':row['external_metric'],
            'metric':row['metric'],'wall_seconds':row['wall_seconds'],'status':row['status'],
            'input_sha256':row['input_sha256'],'runtime_sha256':summary['runtime_code_sha256'],'archive_sha256':row['zip_sha'],
            'primary_split_id':summary['validation']['id'],'validation_kind':summary['validation']['kind'],
            'full_refit':summary['full_refit_done'],'champion_family':summary['champion_recipe']['model_family'],
            'final_components':[c['recipe']['model_family'] for c in json.loads((run/'final_recipe.json').read_text())['components']],
            'qwen_calls':summary['qwen_calls'],'qwen_tokens':summary['qwen_tokens'],
            'strategies':[r['research_strategy'] for r in experiments if r.get('research_strategy')],
            'failed_experiments':[{'id':r['experiment_id'],'phase':r.get('phase'),'status':r['status'],'message':r.get('message')} for r in experiments if r['status']!='ok'],
            'challenger':summary.get('automl_challenger'),
            'challenger_total_worker_seconds':sum(r['runtime_sec'] for r in ag),
            'challenger_peak_memory_mb':max([r['peak_memory_mb'] for r in ag],default=0),
            'challenger_experiments':[{'phase':r.get('phase'),'status':r['status'],'metric':r.get('metric_value'),
                'seconds':r['runtime_sec'],'policy':r.get('challenger_policy')} for r in ag]})
    final=json.loads((out/'final-static-gate.json').read_text())
    report={'stage':'bounded experiment completed; no promotion to stable','observations':details,
        'summary':paired['summary'],'final_artifact':final,
        'metric_scope':'Frozen prototype only, not the corrected final artifact',
        'post_freeze_corrections':['Explicit CPU parameters for AG RF/XT/GBM/CAT',
            'Retain all three main finalists; challenger uses separately funded fourth slot',
            'Use existing research_evidence costs rather than duplicate costs in Qwen action menu',
            'Preflight official setuptools 80.9.0 reload dependency after observed pkg_resources failure'],
        'final_corrected_artifact_full_ml_tested':False,'new_full_ml_runs':4,
        'selection_warning':'Two reused development cases, one seed, stochastic Qwen, prototype RF/XT CPU cap was not strict on 12-thread host; no causal/private improvement claim'}
    (out/'results-detail.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# Ограниченный эксперимент AutoML + стратегический Qwen','',
        '**Статус: эксперимент завершён; замена стабильного решения не рекомендована на основании этих данных.**','',
        'Таблица относится к замороженному прототипу ZIP `7644890c…`, runtime `4588cb3f…`. Финальный исправленный ZIP имеет другой SHA и прошёл unit/static проверки, но не дополнительное полное ML-сравнение: лимит 4 запусков исчерпан.','',
        '| Задача | Метрика | До | Прототип после | Изменение качества | Время до/после |','|---|---|---:|---:|---:|---:|']
    for pair in paired['summary']['pairs']:
        lines.append(f"| {pair['case']} | {pair['metric']} | {pair['reference_score']:.6f} | {pair['candidate_score']:.6f} | {pair['relative_gain']*100:+.2f}% | {pair['reference_seconds']:.1f} / {pair['candidate_seconds']:.1f} с |")
    lines+=['','Изменение ориентировано на качество: для RMSE меньше лучше; для Macro F1 больше лучше. Это внешние development-метрики, не баллы Boosters.','',
        '## Что выполнено','',
        '- Отдельная ветка и воспроизводимый engineering baseline `91777bb`; существующие ускорения, SearchController, research_evidence и защиты воркеров сохранены.',
        '- AutoGluon 1.3.1 как отдельный CPU challenger через существующие Executor, worker, Judge, frozen controls и refit; без нейросетей, скачивания весов, нового HPO-контура и публикации неподтверждённого кандидата.',
        '- Выбранные модели, итерации и веса фиксируются после поискового split; на controls новый ансамбль не подбирается. Feature generator обучается только на train. Группы и дубликаты проверяются; time пропускается.',
        '- Стратегические направления выбираются в тех же вызовах Qwen; доступны только разрешённые TaskSpec/DSL действия, ошибки возвращают обычный deterministic fallback.',
        '- После заморозки исправлены RF/XT n_jobs=-1, вытеснение третьего основного финалиста и дублирование cost payload. Эти поправки отражены только в финальном артефакте и unit/static проверках.','',
        '## Причины и ограничения результатов','',
        'На регрессии прототип проиграл примерно 3.66%. AG обучился на primary и двух controls, но был отклонён по robust utility. Baseline выбрал подтверждённый CatBoost+LightGBM blend; прототип — одиночный CatBoost. Qwen создал другие формулы и другую траекторию TPE, поэтому изолировать влияние одного изменения нельзя. Прототип также заменял третьего main-финалиста challenger; финальная версия сохраняет этот слот. Нет данных, что сама эта поправка возвращает потерянное внешнее качество.','',
        'На классификации AG выиграл внутренний robust selection (0.93503 против 0.93336 у лучшего основного CatBoost), но финальный refit упал при загрузке, до нового обучения: `ModuleNotFoundError: pkg_resources`. В локальном AG-окружении отсутствовал официальный `setuptools==80.9.0`. С ним существующий артефакт успешно загружен без обучения; финальная версия проверяет зависимость заранее. Прототип остался на подтверждённом fold-checkpoint, тогда как baseline завершил full refit. Следовательно, результат не доказывает, что AG после успешного полного refit лучше или хуже.','',
        '## Фактические расходы и ошибки','']
    for r in details:
        if r['arm']!='candidate':continue
        lines.append(f"- {r['case']}: AG worker time {r['challenger_total_worker_seconds']:.2f} с; peak worker RSS {r['challenger_peak_memory_mb']:.1f} MiB; AG выбран: {r['challenger'].get('selected',False)}; ошибок всех экспериментов: {len(r['failed_experiments'])}; Qwen calls {r['qwen_calls']}.")
    lines+=['','Общий AG резерв ограничен min(1800 с, 30% полного бюджета, свободное время после main finalization), включая confirmation/refit; это часть общего дедлайна. Гарантия RF/XT CPU была неполной в прототипе, поэтому его время нельзя объявлять честным speedup при строгих 8 CPU. Финальный явный CPU clamp проверен unit-тестами.','',
        '## Проверки и риски','',
        '58 целевых тестов финального исходника прошли; отдельно сохранены 87 compatibility-проверок предыдущего снимка. Четыре полных запуска выполнены без повторов; все input/split/scorer/archive SHA сохранены в JSON. Финальный ZIP плоский, requirements побайтно официальный, ключей и runtime installers нет. Предыдущие защищённые ZIP сохранены побайтно.','',
        'Новая копия threshold tuning, OOF, blending, AdaptiveTPE и stacking не добавлялась. Для P3 не найден отдельный измеренный недостаток, оправдывающий изменение; для P4 нет ресурса и независимого подтверждения пользы.','',
        'Риски: 2 знакомых development-задачи и один seed; стохастичность Qwen; неполный CPU cap в прототипе; финальные исправления не имеют полного ML-парного теста; нет прогона на полном образе организаторов/3 часа/приватных данных; частный AG accessor извлечения весов закреплён на 1.3.1 и при несовместимости должен завершаться без порчи checkpoint.','',
        f"Финальный артефакт: `submission_quality_strategy_experimental.zip`, SHA256 `{final['archive_sha256']}`; runtime `{final['runtime_code_sha256']}`. На Boosters ничего не отправлялось."]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['stage'],'pairs':paired['summary']['win_tie_loss'],'final_sha256':final['archive_sha256']}))

if __name__=='__main__':main()
