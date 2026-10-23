"""Publish the already-selected archive and compile executed before/after evidence.

This script has no competition upload or runtime tuning operation.
"""
from __future__ import annotations
import csv,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.utils import code_digest
from scripts.benchmark_suite import file_hash
from scripts.report_performance_v2 import paired_effects,stats,resource_stats

D=ROOT/'reports/performance_v2'


def completed(name):
    value=json.loads((D/(name+'.json')).read_text())
    if value.get('status')!='completed':raise RuntimeError('Campaign is not complete: '+name)
    return value


def fmt(value):return '—' if value is None else f'{value:.9g}'
def pct(value):return '—' if value is None else f'{100*value:+.3f}%'


def main():
    freeze=json.loads((ROOT/'benchmarks/performance_v2/final_freeze.json').read_text())
    source=freeze['sources']['candidate'];zip_sha=freeze['archives']['candidate'];version=freeze['candidate_version'];label=version.upper()
    final=completed('final-lockbox-live');broad=completed('p3h-broad-120')
    followup=completed(version+'-annotation-followup');groups=completed('additional-group-diagnostic')
    openml=completed('openml-restored-120') if (D/'openml-restored-120.json').exists() else None
    large_text=json.loads((D/'large_text_runtime.json').read_text()) if (D/'large_text_runtime.json').exists() else None
    if large_text and large_text.get('status')=='running':raise RuntimeError('Large-text operational QA is incomplete')
    if code_digest()!=source:raise RuntimeError('Working runtime changed after final freeze')
    if {arm:row['zip_sha'] for arm,row in final['versions'].items()}!=freeze['archives']:
        raise RuntimeError('Final campaign archive mismatch')
    if final['evaluator_fingerprints']!=freeze['evaluator_fingerprints']:
        raise RuntimeError('Final scorer differs from pre-opening freeze')
    for name,expected in freeze['evaluator_fingerprints'].items():
        if file_hash(ROOT/name)!=expected:raise RuntimeError('Scorer changed after opening: '+name)
    for path,expected in freeze['gates'].items():
        if file_hash(ROOT/path)!=expected:raise RuntimeError('Frozen verification proof changed: '+path)
    final_pairs=paired_effects(final['runs']);final_stats=stats(final_pairs)
    if final_stats['distinct_datasets']<8:raise RuntimeError('Fewer than eight successfully paired final datasets; preserve observations and resolve/report availability before release')
    tests=json.loads((ROOT/'reports/test-state.json').read_text())
    if tests.get('status')!='passed' or tests.get('source_sha256_after')!=source:
        raise RuntimeError('Latest complete test gate differs from final source')
    from scripts.package_submission import build
    manifest=build()
    if manifest['sha256']!=zip_sha:raise RuntimeError('Root package is not byte-identical to the frozen final candidate')
    matrix=json.loads((D/'benchmark_matrix.json').read_text())
    research_counts=dict(Counter(row['status'] for row in matrix['runs']))
    research_cases=len({row['case'] for row in matrix['runs'] if row.get('training_rows')})
    panels=[('development_P3h','p3h-broad-120',broad),
        ('additional_group_P3h','additional-group-diagnostic',groups),
        ('pre_fix_P3i','p3i-annotation-followup',completed('p3i-annotation-followup')),
        ('live_followup_'+label,version+'-annotation-followup',followup),
        ('final_unseen_'+label,'final-lockbox-live',final)]
    if openml:
        panels.append(('post_freeze_OpenML_'+label,'openml-restored-120',openml))
    for name in ['budget-official-regression-10500','budget-official-classification-10500']:
        panels.append((name.removeprefix('budget-')+'_P3h',name,completed(name)))
    detailed=[];family=[];summaries={};resources=[]
    for phase,name,value in panels:
        pairs=paired_effects(value['runs'])
        for comparison_label in sorted({p['comparison'] for p in pairs}):
            summaries[phase+' / '+comparison_label]=stats([p for p in pairs if p['comparison']==comparison_label])
        for pair in pairs:
            mode=pair['comparison'].split(' / ')[-1]
            same=[r for r in value['runs'] if r['case']==pair['case'] and r['seed']==pair['seed']]
            if pair['comparison'].startswith('candidate vs baseline'):
                old=next(r for r in same if r['arm']=='baseline' and r['mode']==mode)
                new=next(r for r in same if r['arm']=='candidate' and r['mode']==mode)
            else:
                old=next(r for r in same if r['mode']=='deterministic')
                new=next(r for r in same if r['mode']==pair['comparison'].split(' vs ')[0])
            row={'panel':phase,'campaign':name,**pair}
            for prefix,observed in [('before',old),('after',new)]:
                for key in ['source_sha','zip_sha','budget_seconds','wall_seconds','cpu_seconds','peak_memory_mb',
                    'first_trained_checkpoint_seconds','full_refit','fit_success','control_folds_completed',
                    'champion_family','qwen_calls','qwen_fits','qwen_accepted','qwen_tokens','split_sha256','input_sha256']:
                    row[prefix+'_'+key]=observed.get(key)
                runtime_summary=Path(observed['run_dir'])/'summary.json' if observed.get('run_dir') else None
                row[prefix+'_runtime_metric']=json.loads(runtime_summary.read_text()).get('metric') if runtime_summary and runtime_summary.exists() else None
            detailed.append(row)
        for arm,mode in sorted({(r['arm'],r['mode']) for r in value['runs']}):
            selected=[r for r in value['runs'] if r['arm']==arm and r['mode']==mode]
            resources.append({'panel':phase,'arm':arm,'mode':mode,**resource_stats(selected)})
            for selected_family,count in Counter(r.get('champion_family') for r in selected if r['status']=='trained_success').items():
                family.append({'panel':phase,'arm':arm,'mode':mode,'selected_family':selected_family,'count':count,
                    'interpretation':'Observed selected champion, not an isolated causal family contribution'})
    def write_csv(path,rows):
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader()
            writer.writerows({key:json.dumps(value,ensure_ascii=False,sort_keys=True) if isinstance(value,(dict,list)) else value
                              for key,value in row.items()} for row in rows)
    write_csv(D/'before_after.csv',detailed);write_csv(D/'family_contribution.csv',family)
    (D/'before_after.json').write_text(json.dumps({'note':'Measured source versions remain separate. Seeds within a dataset are not independent datasets. No platform score.',
        'statistics':summaries,'resources':resources,'pairs':detailed},indent=2)+'\n')
    lines=['# Полная таблица до / после','',
        'Фактические локальные оценки. Положительный относительный эффект означает улучшение; допуск ничьей 0.1%. '
        f'Указаны разные замороженные версии: разработка P3h, финальный контроль {label}. '
        'Ошибки и недоступные оценки не превращены в нули. RMSE/MAE/MAPE/logloss минимизируются; '
        'F1/ROC AUC максимизируются. Сырые разнородные метрики не усредняются.','',
        '| Панель | Задача | Seed | Метрика | Сравнение | До | После | Эффект | Время до / после, с |',
        '|---|---|---:|---|---|---:|---:|---:|---:|']
    for row in detailed:
        lines.append(f"| {row['panel']} | {row['case']} | {row['seed']} | {row['metric']} | {row['comparison']} | "
            f"{fmt(row.get('baseline_score'))} | {fmt(row.get('candidate_score'))} | "
            f"{pct(row.get('relative_gain')) if 'relative_gain' in row else row['outcome']} | "
            f"{fmt(row.get('baseline_seconds'))} / {fmt(row.get('candidate_seconds'))} |")
    lines+=['','CSV содержит SHA исходников/архива/входов/split, CPU, память, refit и фактический вклад Qwen. '
        'Выбор семейства в family_contribution.csv является наблюдением, а не доказательством его причинного вклада.','']
    (D/'before_after.md').write_text('\n'.join(lines))
    if openml:
        openml_manifest=json.loads((ROOT/'benchmarks/performance_v2/openml_followup_manifest.json').read_text())
        by_case={row['case']:row for row in openml_manifest['cases']}
        openml_lines=['# OpenML: дополнительная внешняя проверка','',
            'После восстановления доступа получены те же шесть задач, которые были объявлены до разработки. '
            'Замороженные original HEAD и P3j, deterministic, seeds42/53, cap120с, 8CPU/8192MiB. '
            'Seed42 выбирает исходный repeat0/fold0, seed53 — repeat0/fold1. '
            'Внешняя метрика этого заранее объявленного исследования — Macro F1 (больше лучше), '
            'а не утверждение о воспроизведении всей AMLB или исходной OpenML accuracy. '
            'После просмотра оценок алгоритм не менялся. Это отдельная панель от десяти финальных BeyondArena-задач.','',
            '| Task / данные | Seed | До F1 | После F1 | Эффект | Статусы до / после | Время до / после, с |',
            '|---|---:|---:|---:|---:|---|---:|']
        for pair in paired_effects(openml['runs']):
            record=by_case[pair['case']]
            openml_lines.append(f"| [{record['task_id']}: {record['dataset']}]({record['source_url']}) | {pair['seed']} | "
                f"{fmt(pair.get('baseline_score'))} | {fmt(pair.get('candidate_score'))} | {pct(pair.get('relative_gain'))} | "
                f"{pair['baseline_status']} / {pair['candidate_status']} | {fmt(pair.get('baseline_seconds'))} / {fmt(pair.get('candidate_seconds'))} |")
        openml_lines+=['','```json',json.dumps(stats(paired_effects(openml['runs'])),indent=2),'```','',
            'Метаданные OpenML буквально указывают лицензию `Public`; она не переименована в CC0/CCBY. '
            'Task IDs, dataset IDs/versions, исходные URL, SHA данных и splits, первичные метаданные и '
            'проверки разбиений сохранены в ../../benchmarks/performance_v2/openml_followup_manifest.json. '
            'Отдельная фиксация до обучения: openml_followup_freeze.json рядом с manifest. '
            'Исторические blocked rows не удалены и не заменены этими поздними результатами. '
            'У hepatitis всего16 внешних строк на fold, поэтому большой относительный эффект может быть '
            'следствием одного предсказания. Два folds не означают два независимых датасета.','']
        (D/'openml_external_validation.md').write_text('\n'.join(openml_lines))
    live=[r for r in followup['runs'] if r['mode']=='live-agent']
    counts={key:sum(r.get(key) or 0 for r in live) for key in ['qwen_calls','qwen_fits','qwen_accepted','qwen_tokens']}
    events=Counter(e.get('status') for r in live for e in r.get('qwen_events',[]))
    qwen_stats=stats(paired_effects(followup['runs']))
    qwen={'source_sha':source,'counts':counts,'events':dict(events),'statistics':qwen_stats,
        'scope':'Six same-source live-versus-deterministic pairs, three datasets, seeds42/53, 120s, max2 HTTP calls; internal acceptance is not external improvement.'}
    (D/(version+'_qwen_summary.json')).write_text(json.dumps(qwen,indent=2)+'\n')
    official=[]
    for name in ['budget-official-regression-10500','budget-official-classification-10500']:
        observed=completed(name);official.extend({'campaign':name,**p} for p in paired_effects(observed['runs']))
    summary={'status':'verified_local_release','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'quality_verdict':'No confirmed universal improvement: final quality is reported as observed, including losses and dummy outcomes; this is the preselected functional release, not a leaderboard-win claim.',
        'runtime_source_sha256':source,'archive_sha256':zip_sha,'archive_bytes':manifest['bytes'],
        'root_file_count':len(manifest['files']),'tests_passed':tests['passed'],
        'tests_seconds':tests['seconds'],'panels':summaries,'resources':resources,
        'final_status_counts':dict(Counter(r['status'] for r in final['runs'])),
        'official_training_holdouts_P3h':official,'live_qwen_selected':qwen,'platform_submissions':0,
        'provenance':'Baseline is original HEAD, not a verified identity of the previously uploaded Boosters solution.',
        'gates':freeze['gates'],'final_resumptions':final.get('resumptions',[]),
        'additional_openml':stats(paired_effects(openml['runs'])) if openml else None,
        'large_text_operational_qa':large_text,
        'research_campaign_count':len(matrix['campaigns']),'research_observed_rows':len(matrix['runs']),
        'research_prepared_case_ids':research_cases,'research_status_counts':research_counts,
        'final_freeze_sha256':file_hash(ROOT/'benchmarks/performance_v2/final_freeze.json')}
    (D/'final_release_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    counts_final=final_stats['outcomes'];counts_dev=stats(paired_effects(broad['runs']))['outcomes']
    lines=['# V2 — итог разработки и проверки','',
        f"Подготовлен локально проверенный `submission.zip`: **{manifest['bytes']} байт**, {len(manifest['files'])} корневых файлов. "
        f"**{tests['passed']} теста** прошли за {tests['seconds']:.2f} с. В Boosters решение не отправлялось. "
        'Публичный балл новой версии неизвестен.','',
        f"Полная матрица: {len(matrix['campaigns'])} кампании, {len(matrix['runs'])} наблюдений, "
        f"{research_cases} разных ID задач с подготовленными входами. Это не число независимых экспериментов. "
        'Все исходы: `'+json.dumps(research_counts,ensure_ascii=False)+'`. Ранний AutoGluon smoke с '
        'неверным source fingerprint сохранён как evaluation_error; ранний factory smoke без декларации '
        'лицензии заблокирован до обучения. Поздние исправленные исследования не заменяют эти ошибки.','',
        '## Изменения','',
        'Добавлены native CatBoost, метрики MAPE/PR-AUC, ограниченные текстовые и категориальные специалисты. '
        'Исправлены групповые/редкие классы, статусы dummy/обученного результата, остановка дочерних процессов, '
        'платформенный бюджет, резерв API/подтверждения/refit и воспроизведённая гонка сигнала уже завершившемуся worker. Реализованы общий OOF, ограниченный ансамбль, '
        'стоимостной поиск и проверяемые гипотезы Qwen. Неудачные универсальные варианты target encoding и '
        'fold averaging не стали значениями по умолчанию. Изменения: `atlas/*.py`, фабрика `scripts/benchmark_suite.py`, '
        'адаптеры, проверка упаковки и Linux. Подробности: P0.md, P1.md, P2.md, P3.md.','',
        'Ключевые файлы: `models.py/features.py/encoding.py` — модели и fold-fitted преобразования; '
        '`validation.py` — группы/время/дубли/редкие классы; `orchestration.py/search.py/hpo.py` — бюджет и поиск; '
        '`ensemble.py` — OOF и ограниченное смешивание; `agent.py/ideas.py` — Qwen и рецепты; '
        '`budget.py/worker.py` — изоляция/остановка; `metrics.py/task.py/submit.py` — метрики и схема ответа. '
        'Все эти файлы находятся в `atlas/`. Offline factory: `scripts/benchmark_suite.py`, `benchmark_data.py`, '
        '`source_adapters.py`; выпуск: `package_submission.py`, `verify_zip.py`, `verify_linux.py`, '
        '`audit_performance_release.py`.','',
        '## Измеренное качество','',
        f"Разработка P3h: 13 задач × 2 seed, {counts_dev.get('win',0)}/{counts_dev.get('tie',0)}/{counts_dev.get('loss',0)} побед/ничьих/проигрышей. "
        'Медиана +0.368%, худшая пара −15.789%; отдельная групповая диагностика имеет просадку −22.123%. '
        f'Эти результаты выполнены на P3h, а не переименованы в {label}. Материальные проигрыши разобраны в development_regressions.md.','',
        f"Финальный ранее не открывавшийся контроль {label} с настоящим Qwen: {final_stats['distinct_datasets']} парных задач, "
        f"{final_stats['paired_observations']} пар, {counts_final.get('win',0)}/{counts_final.get('tie',0)}/{counts_final.get('loss',0)} побед/ничьих/проигрышей; "
        f"медиана {pct(final_stats['median_relative_gain'])}, худшая пара {pct(final_stats['worst_single_relative_gain'])}, "
        f"среднее худшего квартиля {pct(final_stats['worst_quartile_mean'])}, геометрическое отношение {fmt(final_stats['geometric_oriented_ratio'])}. "
        'Все замороженные исходы сохранены. Это локальные метрики, не баллы соревнования.','',
        'Универсальное преимущество на этом финальном контроле не подтвердилось. Архив остаётся '
        'выбранным до открытия данных функциональным кандидатом; он не объявляется победителем '
        'рейтинга или доказанным улучшением на неизвестных private-задачах. Алгоритм после отрицательных '
        'результатов не подстраивался.','',
        'Исходы финальной серии: `' + json.dumps(summary['final_status_counts'],ensure_ascii=False) + '`. '
        'Агрегаты качества условны на успешной паре; ошибки исполнения и неподдерживаемые метрики '
        'остаются отдельными исходами в полной таблице.','',
        ('Отдельная операционная проверка сложного текста с900с: '
         f"{large_text['status']}, фактическое время {large_text['wall_seconds']:.2f}с, "
         f"успешных fits {large_text.get('fit_success')}, full refit {large_text.get('full_refit')}. "
         'Только проверка обучения и CSV на тех же входах; внешний scorer и ответы не использовались. '
         'Она не заменяет исходные dummy/timeout исходы120с и не является новым blind-результатом.'
         if large_text else 'Операционная проверка сложного текста с900с ещё не включена в этот промежуточный отчёт.'),'',
        ('Дополнительная проверка OpenML после восстановления доступа: шесть ранее выбранных задач, '
         'два исходных folds0/1 (seeds42/53), deterministic, 120с. Код не изменялся по результатам. '
         f"Исходы пар: {json.dumps(summary['additional_openml']['outcomes'],ensure_ascii=False)}, "
         f"медиана {pct(summary['additional_openml']['median_relative_gain'])}, "
         f"худшая пара {pct(summary['additional_openml']['worst_single_relative_gain'])}. "
         'Это отдельная панель; она не объединяется с live-Qwen финальным контролем. '
         'Особенно малая hepatitis имеет всего16 внешних строк на fold; её эффект дискретен и неопределён.'
         if openml else 'Дополнительная панель OpenML ещё не включена в этот промежуточный отчёт.'),'',
        '[Все показатели до/после](before_after.md), [полный CSV с затратами и SHA](before_after.csv), '
        '[все исследовательские пары, включая неудачи](all_paired_results.csv), '
        '[все исследовательские запуски](benchmark_dashboard.md), [ресурсы](resource_summary.csv), '
        '[финальный контроль](final_lockbox.md), [разбор проигрышей](final_error_review.md), '
        '[полнота и непроверенные требования](experiment_coverage.md), '
        '[отдельный OpenML](openml_external_validation.md), [его график](openml_before_after.png), '
        '[график PNG](final_before_after.png), [график PDF](final_before_after.pdf).','',
        '## Бюджет и надёжность','',
        'Точный CLI `python3 solution.py task.md` имеет default 10500 с и внешний резерв 300 с. '
        'Работа может естественно завершиться раньше. Реально выполнены caps900/1800/3600/7200/10500 и парные '
        'официальные train-only holdout; наиболее долгие испытания измерены в budget_stress.md. '
        'Непрерывный трёхчасовой soak не выполнен. Проверены 18 сценариев плюс SIGKILL, отказ Qwen и '
        'три естественно завершившихся точных CLI. В сценарии singleton после успешных непревзошедших dummy fits '
        'честно сохранён valid_dummy; он не называется обученным чемпионом. Linux/Python3.12: шесть семейств на '
        'регрессии и классификации, 8CPU/6GiB, сеть отключена, VM без host mounts. '
        'Полный образ организаторов и реальная платформа не проверены.','',
        'В предшествующем P3i live-тесте Adult seed53 был реальный fatal_with_fallback: валидный обученный CSV '
        'сохранился, но full refit не завершился. Отдельный тест воспроизвёл гонку macOS при сигнале уже '
        'завершившемуся worker. После исправления 240 процессов завершены/собраны, пять реальных EPERM '
        'обработаны без сбоя. Живой процесс с запретом сигнала по-прежнему вызывает ошибку. '
        'Провальный P3i запуск сохранён как ошибка, а не переименован в успешное измерение.','',
        '## Вклад Qwen и других backend','',
        f"{label}, шесть реальных A1/A0 пар: {qwen_stats['outcomes'].get('win',0)}/{qwen_stats['outcomes'].get('tie',0)}/{qwen_stats['outcomes'].get('loss',0)}; "
        f"медиана {pct(qwen_stats['median_relative_gain'])}. Фактически {counts['qwen_calls']} вызовов, "
        f"{counts['qwen_fits']} обученных предложений, {counts['qwen_accepted']} внутренних принятий, "
        f"{counts['qwen_tokens']} токенов. Нормализаций аннотаций: {events.get('annotations_normalized',0)}. "
        'Внутреннее принятие не доказывает внешний выигрыш; универсальная польза LLM не установлена. '
        'API latency включена в одинаковый 120-секундный cap.','',
        'Native CatBoost исполняется и выбирается внутренним судьёй на части задач; изолированный ранний '
        'трёхзадачный тест дал три финальные ничьи. Причинный внешний выигрыш всего портфеля от одного CatBoost '
        'не установлен. AutoGluon1.3.1 и FLAML2.3.6 проверены как classical-only внешние эталоны на трёх '
        'задачах, общего преимущества не показали и в runtime не включены. LightAutoML/Torch2.5.1 на этом '
        'macOS x86_64/Python3.12 не проверены из-за отсутствия совместимого официального wheel; это не '
        'эксперимент с нулевым качеством.','',
        '## Данные и воспроизводимость','',
        'Kaggle: пять наборов; BeyondArena: исходные curated splits, pinned revision и лицензии в '
        'benchmarks/manifest_v2.json. Исторические OpenML blocked rows сохранены без оценок. '
        + ('Доступ восстановился23.10.2026; отдельный manifest benchmarks/performance_v2/openml_followup_manifest.json '
        'фиксирует шесть исходных задач/версий, данные и splits с SHA, буквальную лицензию OpenML «Public». '
        'Это обозначение не переименовано в CC0/CCBY. Начальная декларация недоступности не переписана. '
        if openml else 'Свежая проверка доступности и отдельный follow-up сохранены отдельно от исходного manifest. ')+
        'Финальные десять наборов, обе версии, seeds42/53, live-agent, 120с, 8CPU/8192MiB и максимум два '
        'Qwen-вызова зафиксированы до первого скачивания parquet; scorer остался неизменным. '
        'На5g внешний MAPE применён одинаково к обеим версиям. Фактический старый runtime молча выбрал '
        'внутренний RMSE, а P3j — MAPE; обе пары сохранены по общему внешнему scorer. Первоначальное '
        'ожидание unsupported не подтвердилось и не заменяет измеренный trained_success. '
        'Внешние ответы находятся только в родительском оценщике. Никакие CSV/веса/модели/ключи не входят в ZIP.','',
        '## Архив','',
        f"Runtime SHA256: `{source}`.\n\nZIP SHA256: `{zip_sha}`.\n\n"
        'Корневой ZIP побайтно совпадает с замороженным финальным кандидатом. requirements.txt побайтно '
        'совпадает с официальным. Доказательства: final_release_audit.json и ../release-manifest.json.','',
        '## Оставшиеся риски','',
        'Материальные групповые и small-data просадки; четыре dummy на сложном тексте при120с; '
        'нестабильная польза Qwen и его квота/латентность; '
        'ограниченное число задач/разбиений и подбор на development; короткий финальный cap не подтверждает '
        'трёхчасовое качество; неизвестные private-типы/схемы; полный organizer image, настоящий маршрут Qwen '
        'на платформе и физические 32GiB не проверены; LAMA/neural quality отсутствуют. '
        'Точный SHA ранее отправленного решения и его публичный breakdown неизвестны.','',
        '## Три будущие отправки владельца как гипотезы','',
        '1. Проверенный текущий ZIP: native portfolio + внутренне проверяемые Qwen-рецепты. Цель — измерить '
        'фактический public breakdown и связать его с SHA.\n'
        '2. После отдельной одинаково бюджетированной проверки: детерминированная версия того же портфеля. '
        'Гипотеза — меньше потерь от API и недостоверных семантических признаков.\n'
        '3. После новой независимой групповой/small-data панели: более консервативная модельная селекция '
        'и калибровка с учётом числа независимых групп. Гипотеза — меньший хвост проигрышей.','',
        'Вторая и третья версии не объявляются готовыми архивами; отправки автоматически не выполняются. '
        'Следующее исследование — бюджет первого обученного результата на сложном mixed-text и '
        'устойчивость отбора при малом числе групп на новых независимых задачах; ожидаемое направление '
        'пользы — сократить dummy/таймауты и хвост проигрышей. Цена — расходы на preprocessing/pilots, '
        'дополнительные folds/refit и снижение скорости поиска. Численный будущий эффект неизвестен.','']
    (D/'FINAL.md').write_text('\n'.join(lines))
    lock=['# Final unseen panel','',
        'Both archives, all ten cases and the complete protocol were fixed before parquet download. '
        'Original repeats0/1, seeds42/53; live organizer Qwen, max2 calls, 120s including refit; '
        '8CPU/8192MiB. No runtime adaptation after opening.','',
        '```json',json.dumps(final_stats,indent=2),'```','',
        'All raw outcomes remain in final-lockbox-live.json, including mixed-text dummy outcomes. '
        'The original HEAD produced trained predictions on the MAPE task but internally optimized RMSE; '
        'P3j optimized MAPE. The same external MAPE evaluates both predictions. '
        'An interrupted campaign continued only missing trial keys under a separate resume audit; '
        'all15 previously completed observations and the frozen runtime/scorer/protocol were preserved. '
        'The full per-task before/after table and costs are in before_after.md/csv. '
        'Archive identities and pre-opening absence proofs: ../../benchmarks/performance_v2/final_freeze.json. '
        'Opening timestamp and freeze SHA: final_freeze.opened.json beside that file.','']
    (D/'final_lockbox.md').write_text('\n'.join(lock))
    state_path=ROOT/'.bootstrap/performance_v2/session-state.json'
    pending_openml=(ROOT/'benchmarks/performance_v2/openml_followup_manifest.json').exists() and openml is None
    state=json.loads(state_path.read_text());state.update(completed=False,stage='release built; additional validation/documentation/link audit pending',
        updated_at_utc=datetime.now(timezone.utc).isoformat(),root_submission_still_preexisting=False,
        active_campaign='reports/performance_v2/openml-restored-120.json' if pending_openml else None,
        outstanding=(['Finish separately frozen six-task OpenML follow-up; do not tune runtime or erase historical blocked outcomes.'] if pending_openml else [])+['Regenerate campaign/cost reports and plots; update README/CONTRACT/WORKLOG; verify final source/archive hashes, links and secret exclusion, then mark completion.'],final_archive_sha256=zip_sha,final_report='reports/performance_v2/FINAL.md')
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'final_statistics':final_stats,'archive_sha256':zip_sha},indent=2))


if __name__=='__main__':main()
