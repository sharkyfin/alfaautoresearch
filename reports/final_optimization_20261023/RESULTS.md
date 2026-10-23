# ATLAS: финальная оптимизация и конкурентная проверка

**Итог: READY_WITH_RISKS.** Финальный кандидат проверен, подготовлен к отправке; на платформу ничего не отправлялось. Технические ошибки подтверждения исправлены; развитие качества показано на development, но универсальное превосходство не доказано.

Архив: `submission_final_optimized.zip` (основной `submission.zip` после promotion должен совпадать побайтно).
SHA-256: `30e053296722de373df8b82cef26a976df8220a696aa1a6cd36cb173e2b61189`.
Runtime: `e81f82499cb6fc1043297065d3a5edab08f3ac0081a37502db975b1afdcd1902`.
Source commit: `056da92`, ветка `research/final-competitive-validation-20261023`.
Плоский ZIP: 85502 байта, 28 корневых файлов, официальный requirements.txt без изменений, ключей/весов/датасетов внутри нет.

## Что реально выполнено

Ровно 6 начальных полных публичных ML-запусков, 2 адресных development и 1 финальная пара из 2 запусков: всего 10. Initial/targeted — 600 с на вариант, deep — 1800 с. Синтетические unit/component/CLI проверки в это число не подмешиваются. Полный трёхчасовой тест не проводился и не согласовывался. Автоматического продолжения после квоты нет; новые дорогие повторы не запланированы.

Сравнивались одинаковые внешние данные, ID, scorers и лимиты 8 CPU / 16 GiB RSS guard. Локальный memory guard строже production 32 GiB. Внешние ответы находились только в памяти evaluator; child sandbox закрывал raw cache, отчёты и другие runs. Время — фактический wall выполнения, не весь выделенный бюджет. CPU seconds суммированы по дереву процессов; RSS — измеренный пик дерева, возможны пропуски между отсчётами.

Источники: [UCI APS](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), [UCI Bike](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), CC BY 4.0. APS development: 48000/12000, stratified seed42, 170 признаков, positive label pos, Average Precision. Bike development: первые 11122 train / следующие 2781 external; final: первые 13903 / последние 3476. Bike имеет 13 признаков; casual/registered (составляющие target), instant/dteday удалены, timestamp построена из даты/часа. Study metrics — AP и RMSLE; они не являются оригинальной стоимостью ошибок Scania или баллами Boosters.

## Полная таблица development

| Задача | Система / версия | Метрика | Значение | Wall, с | CPU, с | Peak RSS, MiB | Full refit |
|---|---|---|---:|---:|---:|---:|---|
| aps_failure | ATLAS / baseline | average_precision | 0.919807 | 455.8 | 2237.0 | 1133.4 | да |
| aps_failure | AutoGluon reference / baseline | average_precision | 0.916187 | 346.5 | 2023.5 | 924.8 | да |
| aps_failure | CatBoost reference / baseline | average_precision | 0.888392 | 30.8 | 135.9 | 749.7 | да |
| bike_sharing_temporal | ATLAS / baseline | rmsle | 0.327465 | 168.7 | 579.9 | 852.5 | да |
| bike_sharing_temporal | AutoGluon reference / baseline | rmsle | 0.306650 | 21.0 | 130.8 | 655.9 | да |
| bike_sharing_temporal | CatBoost reference / baseline | rmsle | 0.300772 | 11.0 | 48.5 | 359.0 | да |
| aps_failure | ATLAS / candidate | average_precision | 0.921670 | 445.8 | 2087.1 | 1064.6 | да |
| bike_sharing_temporal | ATLAS / candidate | rmsle | 0.310502 | 119.1 | 461.0 | 636.7 | да |

Для AP больше лучше; для RMSLE меньше лучше. Macro F1/accuracy не заменяют здесь основную метрику.

До/после ATLAS: APS AP **0.919807 → 0.921670**, +0.001863, или +0.186 процентного пункта AP. Bike RMSLE **0.327465 → 0.310502**, снижение ошибки **5.18%**. Wall соответственно **455.8 → 445.8 с** и **168.7 → 119.1 с**. Это не доказательство универсального ускорения: менялись выбранные модели и траектории поиска.

На Bike development кандидат по-прежнему хуже AutoGluon (0.306650) и CatBoost (0.300772). Нельзя объявлять ATLAS победителем всей таблицы.

**Происхождение:** baseline — ZIP `036f9036…`, runtime `a99f4f98…`; candidate development — ZIP `bb8c0510…`, runtime `86060520…`. Финальный ZIP `30e05329…` отличается от development только нормализацией inert `strategy.reason` и явным ограничением длины этой аннотации в Qwen prompt. Это способно изменить поведение исследователя, поэтому метрики development не объявляются повторным тестом финального ZIP. Точные полные SHA находятся в state/manifest и JSON запусков.

## Финальное подтверждение на не использованном для выбора архитектуры будущем участке

Протокол и ZIP зафиксированы до этой оценки. Бюджет **1800 с на вариант**, 8 CPU, тот же независимый scorer. Сильный CatBoost выбран как reference потому, что он был лучшим на development. После final код runtime не менялся; новых сравнений по этому holdout не проводилось.

| Задача | Система / версия | Метрика | Значение | Wall, с | CPU, с | Peak RSS, MiB | Full refit |
|---|---|---|---:|---:|---:|---:|---|
| bike_sharing_temporal | ATLAS / final | rmsle | 0.338064 | 318.8 | 1128.4 | 679.3 | да |
| bike_sharing_temporal | CatBoost reference / final | rmsle | 0.344771 | 7.7 | 28.7 | 357.9 | да |

ATLAS снизил RMSLE относительно CatBoost на **1.95%** (абсолютно 0.006707), но затратил примерно **41.2×** больше wall. Это выигрыш качества конкретной пары, а не превосходство по эффективности. Оба завершили full refit; итоговый ATLAS выбрал XGBoost с log-target, без blend и новых формул. 115 записей экспериментов включают confirmation/refit; 92 TPE trials завершены, остановка — стагнация, не дедлайн.

Условные paired bootstrap 95% интервалы, 500 resamples:

| Сравнение | Ориентированный выигрыш | 95% интервал |
|---|---:|---:|
| APS candidate − baseline AP | 0.001863 | [−0.003187; 0.006985] |
| Bike dev baseline − candidate RMSLE | 0.016962 | [0.006237; 0.026181] |
| Bike final CatBoost − ATLAS RMSLE | 0.006707 | [−0.000481; 0.014059] |

Для APS стратифицированы строки, для Bike использованы круговые недельные блоки. Интервалы условны на обученных моделях; не учитывают seed/Qwen/search variability. APS и финальная пара включают ноль: строгого статистического доказательства превосходства здесь нет. Development Bike имеет более убедительный сигнал, но это знакомая задача и один seed.

## Реальные ограничения и внесённые изменения

1. **Исправление протокола валидации.** Старые основные бустеры использовали controls в eval_set/early stopping; AG — в tuning_data. Старые control rows также участвовали в primary train. Теперь controls изолируются до любого поиска; target profile Qwen считается только по поисковому пулу. Установленные на primary iterations (в том числе по bag-member) заморожены. При confirmation CB/LGB/XGB не получают eval_set; AG обучает фиксированные публичные leaf models с X_val/y_val=None и фиксированными весами. Изменение control labels не изменяет предсказания в actual-library integration. Старые внутренние CV-метрики поэтому не сравниваются напрямую с новыми.
2. **Вычисления и моделирование.** Два контрольных окна с одним разрешённым training pool используют один fit, сохраняя две отдельные оценки. Вместо 8 подтверждающих fit для 4 финалистов — 4. На IID примерно 60/20/10/10 train/search/control/control; n<100 честно остаётся без независимых controls. RMSLE первоначально обучается в log1p-пространстве. Добавлен AP-aware weighted probe. AG допускается на заранее изолированной temporal-валидации с проверкой времени/gap; CPU ограничен явно для каждого leaf, full refit использует все известные ответы.
3. **Стратегия существующего поиска.** TPE рассматривает все допустимые семейства, учитывая измеренную стоимость, прирост и надёжность, вместо жёсткого отсечения top-2/top-3. Qwen запрашивается при новой информации/плато и достаточном резерве; compute focus explore/exploit реально влияет на следующие три выбора, сохраняя quarantine. Добавлена статистика отдачи исследовательских направлений. Inert strategy.reason нормализуется так же, как уже нормализовались пояснения к формулам. Исправлены feedback-записи без recipe/runtime и пунктуация positive class. Новой архитектуры исполнения не добавлено.

На APS hist занял 263.2 с до и 163.3 с после; высвободившийся бюджет распределился на LGB/XGB/другие проверки. Число завершённых TPE trials выросло 7 → 12, охвачено больше семейств. Это измерение распределения вычислений; отдельного каузального ablation поиска не было.

Сохранены прежние resource/thread/RLIMIT guards, checkpoints, Recipe DSL, feature groups/ablations, fold-fitted encodings, Independent Judge, threshold/ensemble selection по primary, OOF coverage masks и full refit. В actual traces всех трёх новых ATLAS public-runs: 0 worker errors, 0 control fit-validation rows, disjoint sealed controls, все доступные training rows в refit.

## Что добавили Qwen, AutoResearch и Challenger

- На исходном APS два Qwen ответа отклонены bounded reason guard: 0 обученных Qwen-рецептов. В development-кандидате 1 из 2 запросов дал принятый признак frequency(ck_000), сохранённый в финальном hist. При зафиксированных параметрах primary AP вырос примерно 0.857729 → 0.861723; дальнейшее решение по контролям выбрало 700 iterations. Внешний AP +0.001863 относится ко всему пакету, не изолированному эффекту Qwen.
- На Bike development Qwen не дал принятой модели. На final оба запроса дошли до обучения без отклонения reason guard, но 0 улучшений принято. Финальная модель — TPE XGBoost; положительного отдельного вклада Qwen в эту финальную метрику не доказано.
- AG успешно обучился и получил независимые оценки на обоих control windows на APS и обеих Bike-проверках, включая temporal. Ни в одном новом запуске он не выиграл robust selection. Его измеренные суммарные fit costs: APS 59.5 с, Bike development 18.3 с, Bike final 18.2 с. Поэтому прирост качества от Challenger не заявляется; подтверждены работоспособность, независимость и доступность рационального сравнения.
- Самостоятельный AG использовал более широкий внутренний train и 5-fold bagging на APS; Challenger имеет ограниченную небэговую политику и общий бюджет с исследованием. На temporal reference и агент также имеют разные внутренние горизонты: выбор по меньшему train/другому будущему окну может не совпасть с ранжированием внешнего периода. Это объясняет возможные проигрыши; external labels для принудительного выбора AG не используются.

## Отклонённые усложнения

Новый OOF-stacking не добавлен: после изоляции controls покрытие около 40% IID строк; полный честный nested/OOF контур потребовал бы больше обучения. Нельзя заполнять пропуски target. Новые NN/pretrained-модели и огромный ансамбль не добавлены без подтверждённой CPU-отдачи. Повторный подбор weights/threshold по controls запрещён. Проверенный runtime не заменён переписанной системой. Измеренных отрицательных изменений не скрывали; на development сильные references сохранены в таблице. На final ни одного изменения после оценки не внесено.

## Выполненные проверки и риски

- Финальный source: **386 passed, 8 skipped**, 129.98 с. Отдельно **5 passed** actual AG integration (search → frozen confirmation → all-label refit, плюс GBM/CAT/RF/XT multiclass API). AG код после этой проверки не менялся.
- Настоящий flat bootstrap из финального ZIP выполнен с dev hook 90 с/2 поисковых эксперимента: full refit, sealed controls, schema/IDs/finite output. Проверено отклонение повреждённого source checksum. Это не прогон default 10500 с.
- Static gate: плоские .py + официальный requirements, разрешённые импорты, без installers/ключей/весов/внешних данных. Исторические архивы сохранены и проверены SHA.
- Все 10 публичных benchmark cells завершились успешно с full refit. Detailed runtime trace gate трёх новых ATLAS cells подтверждает реальный frozen protocol в сериализованных jobs/results.

Остаются: один seed и стохастический Qwen; всего два типа реальных задач; нет нового final APS official-test, большого текста/Amazon high-cardinality/Covertype-scale; нет запуска на полном образе организаторов и полного трёхчасового теста; ограниченный портфель AG и pinned private accessor извлечения его весов; finite shortlist всё ещё выбирается по controls, поэтому это validation selection, а не несмещённая test-оценка. Меньший search train — цена независимости, не универсально оптимальная доля. На Bike development ATLAS проигрывает простым references; final выигрыш мал и его CI включает ноль. Ошибки неизвестной private-среды исключить нельзя.

Небольшая диагностическая особенность: поле AG status может оставаться awaiting_independent_controls после завершения; фактическое завершение подтверждается confirmation_folds=2, selected и jobs/results. Это не скрытая незавершённость обучения.

**READY_WITH_RISKS** выбран потому, что формат, модели, независимое подтверждение и refit проверены, а публичное качество конкурентно на этих данных; оснований обещать первое место или отсутствие любого private-сбоя нет. Следующие архитектурные изменения после финальной оценки не имеют достаточно независимых свидетельств отдачи относительно риска и стоимости. Финальный ZIP сохранён, автоматическая отправка не выполнялась.

Команда из распакованного ZIP с Qwen environment организаторов:

```sh
python3 solution.py task.md
```

См. all-results.csv/json, initial/targeted/deep-comparison.json, deep-protocol.json, runtime-trace-gate.json, final-static.json, flat-entry-gate.json, final-manifest.json, state.json и raw test/controller logs. Полный разбор кода — AUDIT.md; протокол — REPRODUCE.md.
