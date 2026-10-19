# ATLAS AutoResearch

Автономный CPU-агент для табличной регрессии и классификации Alfa Hack AutoML.
Статус: разработка, ZIP предварительный до закрытия release gates.

## Запуск

Python 3.12; библиотеки из официального `artifacts/official/requirements.txt`.
Для разработки достаточно подмножества:

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python solution.py /absolute/path/task.md
```

Без аргументов используется `task.md` в cwd, как в официальном baseline.
Относительные CSV-пути разрешаются от каталога Markdown. Для публичных задач
результат `submission.csv` записывается рядом с Markdown.
Никаких установок, загрузок моделей или внешних тренировочных данных в runtime.

Первый CSV содержит безопасный константный прогноз. Улучшения публикуются через
проверенную атомарную замену, поэтому ошибка/timeout/kill worker сохраняют ранее
записанный результат. Оркестратор использует deadline 175 минут; платформенный
предел 180 минут оставляет 5 минут запаса. SIGTERM/SIGINT останавливают workers.

Официальные переменные Qwen: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.
При отсутствии полного набора используется детерминированный поиск. Собственные
ключи других сервисов запрещены. Ключи не передаются ML-workers и не пишутся в
промпты/логи. Доступность Qwen внутри платформы пока не подтверждена.

Разработка с уменьшенным бюджетом и явным режимом:

```sh
.venv/bin/python scripts/run_dev.py path/task.md --budget 360 --threads 8 --mode deterministic --run-dir runs/experiment
.venv/bin/python scripts/run_dev.py path/task.md --budget 60 --mode mock-agent
.venv/bin/python scripts/run_dev.py path/task.md --budget 360 --mode live-agent
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/benchmark_suite.py
.venv/bin/python scripts/package_submission.py
```

`mock-agent` проверяет инфраструктуру; его результаты не являются ablation настоящей Qwen.
Полные входные CSV, модели и ключи в ZIP не входят; пакеты предоставляет платформа.

## Устройство

- TaskSpec объединяет словарь ролей, train/test schema, sample и описание метрики.
- Валидация фиксируется до поиска: stratified/random, forward при подтверждённом
  прогнозировании, group для новых сущностей; точные дубликаты не разделяются.
- Линейные модели, CatBoost, LightGBM, XGBoost, HistGradientBoosting и ExtraTrees.
- Предобработка, frequencies и group aggregates обучаются только на training fold.
- Qwen предлагает ограниченную JSON-рецептуру. Validator запрещает произвольный
  код, импорты, target/ID в feature ops, изменение метрики и схемы валидации.
- Компилятор генерирует Python с выражениями признаков; новый worker запускает
  настоящий fit. Trusted judge рассчитывает метрику по сохранённому holdout.
- Локальный HPO, подтверждение на втором split, ограниченный convex ensemble и
  вероятностные поправки принимаются по исходной метрике.
- Полный refit публикуется только при успехе всех компонентов. Иначе остаётся
  уже проверенный прогноз модели с честной holdout-валидацией.

## Совместимость и ограничения

RMSE, MSE, MAE, R2, RMSLE; accuracy, balanced accuracy; binary/macro/micro/weighted
F1; ROC AUC, logloss, binary average precision. Неизвестная метрика явно
диагностируется и использует RMSE/Macro F1 как fallback, без заявления оптимизации
неизвестной формулы. Multi-target/неоднозначные target и probability schema
отклоняются с UnsupportedTask. Для вероятностей имена multiclass-колонок должны
точно совпадать с исходными метками классов; binary допускает одну колонку.

Это ограниченная DSL, а не sandbox для произвольного Python. Linux worker имеет
RLIMIT_AS, parent проверяет RSS всего дерева и wall-clock; сетевые действия из
рецептуры невозможны. Контейнер/cgroup ограничения дополнительно задаёт платформа.
Локальная проверка идёт на macOS: 12 логических CPU, 16 GiB RAM. Docker пока
недоступен. В macOS LightGBM/XGBoost требуют libomp; в этом dev-окружении применена
ссылка на уже установленную libomp из sklearn. В ZIP никаких dylib нет.

## Артефакты

`CONTRACT.md` — подтверждённый интерфейс и блокеры;
`reports/` — реальные результаты тестов и экспериментов;
`runs/` — локальные recipe.py, splits, predictions, журнал и champion.json;
`submission.zip` — source-only архив с корневым solution.py.
На платформу архив автоматически не отправляется.

## Воспроизведение публичных опытов

Публичные исходные файлы не включаются в git и ZIP. При доступности общих Drive-ссылок:

```sh
.venv/bin/python scripts/download_official.py
.venv/bin/python scripts/prepare_public.py
.venv/bin/python scripts/run_dev.py artifacts/data/regression/task.md --budget 360 --run-dir runs/my-regression
.venv/bin/python scripts/run_dev.py artifacts/data/classification/task.md --budget 360 --run-dir runs/my-classification
```

Сначала проверены 18 фактических fit/predict комбинаций: шесть семейств ×
регрессия, бинарная и многоклассовая классификация. Актуальный итог тестов и
метрик находится в [reports/EVALUATION.md](reports/EVALUATION.md).
Исследовательские прогоны с изменявшимся кодом помечены exploratory;
итоговые сравнения используют копию исходников с фиксированным SHA256.
