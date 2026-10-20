# ATLAS AutoResearch

Автономный CPU-агент для табличной регрессии и классификации Alfa Hack AutoML.
Текущие результаты и открытые проверки: `reports/final-release-audit.json`.
Ключ организаторов подключён; протокол и настоящее обучение по гипотезе Qwen проверены.

## Запуск

Python 3.12; библиотеки из официального `artifacts/official/requirements.txt`.
Для разработки достаточно подмножества:

```sh
"$HOME/.local/bin/uv" venv --python 3.12.14
"$HOME/.local/bin/uv" pip install --python .venv/bin/python -r requirements-dev.txt
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

Платформа задаёт `OPENAI_API_KEY` и `OPENAI_BASE_URL`; `OPENAI_MODEL` необязательна:
по умолчанию используется документированное имя `qwen/qwen3.8-27b`.
Обещанная инструкцией подстановка произвольной модели не сработала на dev-прокси
(HTTP 403), поэтому generic placeholder не используется. При отсутствии адреса или ключа используется
детерминированный поиск. Ключи не передаются ML-workers и не пишутся в промпты/логи.
Локальный доступ к организаторскому Qwen подтверждён; маршрут из оценочного
контейнера можно проверить только на платформе. Дневной бюджет команды — 600 рублей,
сброс в 03:00 МСК. Это отдельный лимит от числа отправок решений.

Разработка с уменьшенным бюджетом и явным режимом:

```sh
.venv/bin/python scripts/run_dev.py path/task.md --budget 360 --threads 8 --mode deterministic --run-dir runs/experiment
.venv/bin/python scripts/run_dev.py path/task.md --budget 60 --mode mock-agent
.venv/bin/python scripts/run_organizer.py path/task.md --budget 120 --max-calls 2 --run-dir runs/my-qwen
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/benchmark_suite.py
.venv/bin/python scripts/package_submission.py
```

`mock-agent` проверяет инфраструктуру; его результаты не являются ablation настоящей Qwen.
Ключ хранится в игнорируемом `.env.organizer.local` с правами `0600`. Dev-wrapper
читает файл без shell/eval; в архив он не входит.

Полные входные CSV, модели и ключи в ZIP не входят; пакеты предоставляет платформа.
Согласно новой инструкции, **все файлы ZIP находятся в корне**. Там лежат
`solution.py`, дополнительные собственные `.py` и неизменённый официальный
`requirements.txt` из Drive-папки. Bootstrap восстанавливает исходный пакет в `/tmp`,
проверяет SHA256 каждого модуля и запускает его. `scripts/verify_zip.py` имитирует
копирование платформой только корневых файлов и команду `python solution.py task.md`.
Прежняя упаковка с вложенным `atlas/` отозвана: она не соответствует этому уточнению.

По отдельному указанию пользователя подключён бесплатный OpenRouter для локальной
разработки. Выбрана `minimax/minimax-m3.1-flash:free`: активный бесплатный endpoint,
JSON-ответы и подходящие опубликованные показатели кода/агентных задач. Это выбор
первого кандидата, а не доказательство превосходства на наших ML-задачах.

```sh
.venv/bin/python scripts/run_openrouter.py path/task.md --budget 120 --max-calls 2 --run-dir runs/my-openrouter
```

Ключ берётся из `OPENROUTER_API_KEY` либо `.env.openrouter.local` (права `0600`,
файл исключён из git). Модель зафиксирована с `:free`, предел цены input/output
принудительно равен нулю; платного fallback нет. MiniMax требует reasoning:
используется low effort и до 4096 output tokens. API-ключ не передаётся workers.
В журнале происхождение гипотез отмечается `openrouter`, а счётчик Qwen остаётся
нулевым. Dev-клиент и файл ключа не входят в ZIP; официальный `solution.py`
самостоятельно не подключается к OpenRouter. Разрешение организаторов на такое
подключение в конкурсном контейнере не получено.

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
Хост разработки — macOS: 12 логических CPU, 16 GiB RAM. Linux-проверка выполнена
в Lima VM без монтирования домашней папки и файлов ключей: 8 CPU, контейнер 6 GiB,
без сети; все шесть семейств обучились на регрессии и классификации. Проверены
RLIMIT_AS, RSS и timeout, валидные CSV после SIGTERM. Подробности: `reports/linux-verification.json`. В macOS LightGBM/XGBoost требуют libomp; в этом dev-окружении применена
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

## Проверка Linux-контейнера при наличии Docker

`Dockerfile.verify` устанавливает утверждённое подмножество пакетов и копирует
**содержимое собранного архива**, а не дерево разработки. Установка выполняется
только при сборке dev-образа; evaluation runtime ничего не устанавливает.

```sh
.venv/bin/python scripts/package_submission.py
docker build -f Dockerfile.verify -t atlas-verify .
docker run --rm --cpus=8 --memory=32g --network=none -v /absolute/task-directory:/task atlas-verify task.md
```

Это проверяет детерминированный offline-путь. Qwen использует единственное разрешённое
сетевое исключение — прокси из окружения платформы. Настройки не сохраняются
в Dockerfile, git или ZIP. Linux smoke с меньшим лимитом памяти не равен
полному воспроизведению организаторского образа и физических 32 GiB.

Каждый `--run-dir` должен указывать на новый пустой каталог. Официальный запуск
сам создаёт уникальный каталог `.atlas/<time>_<suffix>`. Итоговые тесты привязаны
к SHA256 runtime-исходников в `reports/test-state.json`; отсутствие изменений
между началом и концом тестов проверяется автоматически.

Предварительный релиз: 154 теста прошли; 35 stress-сценариев на двух seeds
завершились без неожиданных ошибок. Архив повторно распакован в пустой каталог:
regression и multiclass прошли официальный CLI с реальным обучением и SIGTERM.
Размер, SHA256 и полный список 19 файлов — в
[reports/release-manifest.json](reports/release-manifest.json); проверка релиза —
в [reports/RELEASE_AUDIT.md](reports/RELEASE_AUDIT.md).
Локальные метрики и замеры относятся к указанным в отчёте версиям исходников.
