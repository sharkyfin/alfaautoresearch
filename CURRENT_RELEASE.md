# ATLAS monster

Статус: **READY_WITH_RISKS**. Git-ветка `monster` создана от проверенной финальной версии `e826f9901a1623590d651e5a5b3b1a75acbfaefa`. Исходники этой ветки соответствуют архиву.

Архив для отправки: [monster.zip](monster.zip), 85502 байта, 28 файлов в корне.
SHA-256: `30e053296722de373df8b82cef26a976df8220a696aa1a6cd36cb173e2b61189`.
Runtime digest: `e81f82499cb6fc1043297065d3a5edab08f3ac0081a37502db975b1afdcd1902`.
Проверка целостности: `shasum -a 256 -c monster.zip.sha256`.

[Полный ML-отчёт](reports/final_optimization_20261023/RESULTS.md), [таблица результатов](reports/final_optimization_20261023/all-results.csv), [release manifest](reports/release-manifest.json).

Исходная версия прошла 386 unit-тестов и 5 actual AutoGluon component tests; выполнены 10 публичных ML-запусков с полным refit. Последняя операционная проверка самого архива: 11 адресных unit-тестов, четыре Linux-пробы (16 успешных fit, 0 ошибок workers, валидные обученные CSV), включая одновременно pids.max=30 и унаследованный RLIMIT_AS=8 GiB. Короткий flat-bootstrap прогон отдельно подтвердил full refit и checksum guard. [Подробности и ограничения](reports/monster-last-check/RESULTS.md), [машиночитаемый итог](reports/monster-last-check/final.json).

Архив сохранён побайтно: новый runtime не добавлялся. Последний экспериментальный Qwen-кандидат не включён из-за ухудшения development-результата. Весь прежний финальный код доступен в истории ветки.

После распаковки, с API-параметрами организаторов в окружении:

```sh
python3 solution.py task.md
```

Runtime использует до 10500 секунд с резервом до платформенного лимита 10800 секунд, 8 CPU и worker memory guard 24576 MiB. CSV записывается по контракту задачи. Полного запуска в образе организаторов и проверки скрытых задач не было; ненулевой балл не гарантирован. На платформу ничего не отправлялось.
