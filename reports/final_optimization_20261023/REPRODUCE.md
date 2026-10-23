# Протокол и воспроизведение

Исходники: ветка `research/final-competitive-validation-20261023`, исходный commit финального runtime `056da92`.
Worktree текущего исследования: `.bootstrap/final_optimization/worktree` относительно основного проекта.
Production-запуск из распакованного ZIP:

```sh
python3 solution.py task.md
```

Нужны Python 3.12, официальный requirements.txt и переменные Qwen, выданные организаторами. Установка пакетов и скачивание весов во время решения не выполняются. Production default 10500 с оставляет запас до лимита 10800 с; полный трёхчасовой запуск не проводился.

## Данные и защита оценки

- [UCI APS, dataset 421](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), CC BY 4.0. Архив SHA256 `5504d0402f54faaf97ac0ca085a621645763f5cfea2eb29c592b057d43d4db89`.
- [UCI Bike Sharing, dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), CC BY 4.0. Удалены casual и registered (их сумма равна target), а также instant и dteday; добавлена timestamp из даты и часа. Фактически 17379 строк hour.csv.
- APS development: 48000/12000, stratified seed42, Average Precision положительного класса pos. Это study metric, а не оригинальная асимметричная стоимость ошибок Scania.
- Bike development: первые 11122 строки train, следующие 2781 — внешняя оценка; последние 3476 зарезервированы для финальной пары. RMSLE — выбранная метрика исследования count-регрессии.
- Финальная пара: первые 13903 строки Bike train, последние 3476 external. Её метрики не используются для изменения кода или новых гиперпараметров.
- Truth хранится только в памяти родительского evaluator, нет файла с test labels в доступном агенту task. Sandbox запрещает чтение исходных кешей, reports, ключей с диска и других runs. ATLAS получает только разрешённые task-файлы и Qwen API. Референсам сеть закрыта.
- Все системы имеют одинаковые внешние train/test, input SHA, scorer и лимиты 8 CPU/16 GiB RSS guard. Локальный guard строже production 32 GiB. Фактическое время может быть меньше бюджета. Общий хост не является официальным Docker-образом организаторов.

## Запуски

Начальные 6 полных запусков: 2 задачи × ATLAS/AG/CatBoost, 600 с каждый. Адресные: 2 запуска ATLAS с development ZIP bb8c0510…, 600 с. Одна финальная пара: финальный ZIP 30e05329…, ATLAS и CatBoost, 1800 с на вариант.

```sh
# Выполнялись из worktree; PYTHON указывает на локальное окружение с approved AG 1.3.1.
"$PYTHON" scripts/competitive_compare.py --stage initial --budget 600 --phase development --report "$REPORTS/initial-comparison.json"
"$PYTHON" scripts/competitive_compare.py --stage targeted --archives "candidate=$PROJECT/submission_final_candidate.zip" --modes live-agent --cases aps_failure,bike_sharing_temporal --budget 600 --phase development --report "$REPORTS/targeted-comparison.json"
"$PYTHON" scripts/competitive_compare.py --stage deep --archives "final=$PROJECT/submission_final_optimized.zip" --modes live-agent,catboost --cases bike_sharing_temporal --budget 1800 --phase final --report "$REPORTS/deep-comparison.json"
```

PROJECT — основной каталог проекта, REPORTS — reports/final_optimization_20261023. В текущем ledger повторные initial/deep запуски специально запрещены; репликация требует нового независимого исследования с новым ledger и report, а не обхода лимита текущего этапа. Dev-адаптеры содержат абсолютный путь к этому PROJECT; при переносе его нужно настроить.

AG 1.3.1 reference: классический портфель GBM×3/CAT/RF/XT, 5-fold bagging для APS, chronological holdout без bagging для Bike, без stacking/нейросетей/весов. Inner tuning разрешено, external labels недоступны. CatBoost reference: 6000 iterations/depth7/lr.05/l2=3, внутренний early stopping, log-target RMSLE, полный refit по найденным iterations.

## Проверки и происхождение

- Development ZIP: bb8c0510e5af13b0097874e7b251e6ea3b8a98b8c94a3b74e0bc5597795f3deb, runtime 860605202bd4011f95839287254dc79b92fce448e1323712cdc27d208e160862.
- Final ZIP: 30e053296722de373df8b82cef26a976df8220a696aa1a6cd36cb173e2b61189, runtime e81f82499cb6fc1043297065d3a5edab08f3ac0081a37502db975b1afdcd1902.
- Единственная runtime-разница final vs development: нормализация inert strategy.reason и явная длина пояснения в Qwen prompt. Это может изменить поведение исследователя; development-метрики нельзя приписывать окончательному ZIP без этой оговорки.
- Final frozen suite: 386 passed, 8 skipped; actual AutoGluon component integration отдельно 5 passed. Skips — зависимости/архивные вариации; они не выдаются за исполненные проверки.
- Flat CLI integration действительно запускает solution.py из ZIP, но dev import hook ограничивает тест 90 сек/2 поисковых эксперимента. Проверены полное обучение, sealed controls, схема/ID/конечность output и отклонение испорченного source checksum. Это не тест полного production budget.
- Bootstrap CIs: 500 paired resamples, stratified rows для APS и circular weekly blocks для Bike. Они условны на данных двух обученных моделей, не включают вариацию seed/Qwen/полного поиска. Нельзя интерпретировать их как доказательство универсального выигрыша.
- Основной итог и все численные результаты находятся в RESULTS.md и соответствующих JSON/CSV. Никаких platform upload или auto-continuation не выполнялось.
