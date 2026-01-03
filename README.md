# OCR: вопросы с изображений

## Установка

1. Нужен Python 3.12+.

2. Создайте виртуальное окружение и установите зависимости:

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. Убедитесь, что установлен tesseract и языки `rus`/`eng`.

## Использование

Извлечь текст из центра изображений и сохранить в JSON:

```
python scripts/extract_questions.py ./data/photos --out data/questions.json
```

Параметры:
- `--crop left,top,right,bottom` — настройка обрезки (проценты).
- `--lang` — языки tesseract (по умолчанию: `rus+eng`).

## Загрузка новых картинок из публичного Telegram-канала

Если нет возможности создать бота, можно брать изображения из публичной веб-версии канала (`https://t.me/s/...`).
Скрипт хранит `last_seen_id` и при следующем запуске тянет только новые картинки.
Для бэктфила по частям используется `backfill_before_id`, чтобы продолжать со
старых постов при следующих запусках.

Пример разового запуска:

```
python3 scripts/fetch_channel_images.py --channel vopros_dna --out-dir data/photos
```

Частичный бэктфил с продолжением в следующих запусках:

```
python3 scripts/fetch_channel_images.py --channel vopros_dna --out-dir data/photos --backfill --max-pages 50
```

Ограничение скорости запросов:

```
python3 scripts/fetch_channel_images.py --channel vopros_dna --out-dir data/photos --page-sleep 1.0 --download-sleep 0.3
```

Логи пишутся в консоль и файл с датой (UTC время в каждой строке),
например `data/telegram_fetch_YYYY-MM-DD.log`. Логи старше 30 дней удаляются
автоматически. Можно использовать `{date}` в пути. Базовый путь можно задать так:

```
python3 scripts/fetch_channel_images.py --channel vopros_dna --out-dir data/photos --log-file data/telegram_fetch.log
```

Пример ежедневного запуска в полночь через cron:

```
0 0 * * * /opt/ocr/.venv/bin/python /opt/ocr/scripts/fetch_channel_images.py --channel vopros_dna --out-dir /opt/ocr/data/photos --state-file /opt/ocr/data/telegram_state.json
```

Скрипт обновляет манифест `data/telegram_manifest.json` (метаданные постов)
и индекс `data/daily_index.json` (статистика загрузки).
Пути можно переопределить через `--manifest-file` и `--index-file`.

## Формат JSON и индекс

OCR скрипт создаёт отдельный JSON на каждую дату (UTC), а скрипт модерации
обновляет флаги цензуры через Mistral AI.
По умолчанию файлы создаются как `data/questions_YYYY-MM-DD.json`.
Внутри для каждой картинки есть `number`, `datetime`, `filename`, `text`,
`tesseract_text`, `easyocr_text`, `ocrspace_text`, `mistral_text`,
`llm_validated`, `human_validated`, `is_correct`, `tg_message_id`, `tg_datetime_utc`
и поля цензуры `is_sexual`,
`is_profanity`, `is_politics`, `is_insults`, `is_threats`, `is_harassment`,
`is_twitch_banned` (изначально `true`). Время из Telegram пишется в UTC+0.
Поле `text` заполняется только когда `is_correct=true` (совпали минимум два OCR‑варианта),
иначе `text` остаётся пустым, а варианты сохраняются в `*_text`.
Дополнительно создаётся индекс `data/daily_index.json` со списком дат,
путями к JSON и статистикой по загрузке и OCR.
Поля индекса: `date`, `json_path`, `download_success`, `download_failed`,
`downloaded_count`, `downloaded_new_count`, `download_failed_count`,
`download_failed_new_count`, `ocr_success`, `ocr_failed`, `ocr_failed_count`,
`ocr_extracted_phrases`.

Пример JSON с распознанным текстом:

```json
[
  {
    "number": 123,
    "datetime": "2024-10-07 10:10:05",
    "filename": "photo_123@07-10-2024_10-10-05.jpg",
    "text": "Какая ваша любимая книга?",
    "tesseract_text": "Какая ваша любимая книга?",
    "easyocr_text": "Какая ваша любимая книга?",
    "ocrspace_text": "",
    "mistral_text": "",
    "llm_validated": false,
    "human_validated": false,
    "is_correct": false,
    "tg_message_id": 5511,
    "tg_datetime_utc": "2024-10-07 10:10:05",
    "is_sexual": true,
    "is_profanity": true,
    "is_politics": true,
    "is_insults": true,
    "is_threats": true,
    "is_harassment": true,
    "is_twitch_banned": true
  }
]
```

Пример `daily_index.json`:

```json
{
  "2024-10-07": {
    "date": "2024-10-07",
    "json_path": "data/questions_2024-10-07.json",
    "download_success": true,
    "download_failed": false,
    "downloaded_count": 10,
    "downloaded_new_count": 2,
    "download_failed_count": 0,
    "download_failed_new_count": 0,
    "ocr_success": true,
    "ocr_failed": false,
    "ocr_failed_count": 0,
    "ocr_extracted_phrases": 10
  }
}
```

## OCR изображений из каталога

Скрипт берёт изображения из каталога, использует манифест Telegram (если есть)
и пишет JSON по датам (UTC). Также обновляет `data/daily_index.json`.
Он параллельно распознаёт каждый файл через Tesseract и EasyOCR, при расхождении
добавляет OCR.space, и если совпадают минимум два варианта — записывает `text`
и ставит `is_correct=true`. Если все три варианта различаются, он запрашивает
Mistral и сохраняет результат в `mistral_text`. После этого, если совпадают
минимум два из четырёх вариантов, `text` заполняется и `is_correct=true`;
если все четыре различаются — `text` остаётся пустым.

Пример запуска:

```
python3 scripts/ocr_images.py ./data/cropped
```

Переопределить путь можно через `--out-json` (поддерживает `{date}`).
Индекс можно переопределить через `--index-file` (или `DAILY_INDEX_FILE`).

Логи пишутся в консоль и файл с датой (UTC время в каждой строке),
например `data/ocr_YYYY-MM-DD.log`. Логи старше 30 дней удаляются автоматически.

## Предобработка изображений

Скрипт проверяет нижнюю часть картинки через Tesseract на наличие
`vopros` / `vopros.dna` / `vopros_dna`, а также `.dna` / `_dna` / `dna`.
Если текст найден, картинка обрезается сверху и снизу на 1/4 плюс
дополнительно 23 пикселя сверху, и сохраняется в `data/cropped`
под тем же именем. Если текст не найден, копия изображения
сохраняется в `data/not_question`. Оригиналы остаются в `data/photos`.
Перед обработкой все миниатюры `*_thumb.*` удаляются.

Пример запуска:

```
python3 scripts/preprocess_questions.py ./data/photos
```

## OCR через OCR.space

Скрипт делает OCR через OCR.space и пишет JSON по датам (UTC).
Параметр `OCRSPACE_LANGUAGE` принимает 3‑буквенные коды (`rus`, `eng`)
или `auto` для autodetect (только при `OCRSPACE_OCR_ENGINE=2`).

Пример запуска:

```
python3 scripts/ocr_space.py ./data/cropped
```

## OCR через EasyOCR

Скрипт делает OCR через EasyOCR и пишет JSON по датам (UTC).
Языки задаются через `EASYOCR_LANGS` или флаг `--langs`, GPU/CPU —
через `EASYOCR_GPU` или `--gpu/--cpu`.

Пример запуска:

```
python3 scripts/ocr_easyocr.py ./data/cropped
```

## Модерация JSON через Mistral

Скрипт обновляет только флаги цензуры в указанных JSON файлах.
Он не меняет `is_correct` и не выполняет дополнительные OCR‑фоллбэки.

Пример запуска:

```
python3 scripts/moderate_json.py data/questions_2024-10-07.json --images-dir data/cropped
```

Можно обработать папку:

```
python3 scripts/moderate_json.py --input-dir data
```

Размер батча задаётся через `MISTRAL_BATCH_SIZE` (по умолчанию: 10).
Логи пишутся в консоль и файл с датой, например `data/mistral_YYYY-MM-DD.log`.
При `--input-dir` скрипт автоматически пропускает файлы `*_ai_ocr.json`
и `*_ocr_failed_*.json`.

Перед запуском задайте API-ключ:

```
export MISTRAL_API_KEY=ваш_ключ
```

Либо положите ключ в `.env` (см. `.env.example`):

```
# Mistral API key for moderation requests.
MISTRAL_API_KEY=ваш_ключ
# Mistral model name.
MISTRAL_MODEL=mistral-small-latest
# Mistral batch size for moderation.
MISTRAL_BATCH_SIZE=10
# Mistral request timeout in seconds.
MISTRAL_TIMEOUT=60
# Sleep between Mistral requests (seconds).
MISTRAL_REQUEST_SLEEP=10
# OCR.space API key.
OCRSPACE_API_KEY=your_key_here
# OCR.space API base URL.
OCRSPACE_API_URL=https://api.ocr.space/parse/image
# OCR.space language (3-letter code like rus, eng; or auto for engine 2).
OCRSPACE_LANGUAGE=rus
# OCR.space OCR engine (1 or 2).
OCRSPACE_OCR_ENGINE=2
# OCR.space request timeout in seconds.
OCRSPACE_TIMEOUT=60
# Sleep between OCR.space requests (seconds).
OCRSPACE_REQUEST_SLEEP=10
# Crop for external OCR (left,top,right,bottom).
EXTERNAL_OCR_CROP=0.08,0.18,0.08,0.20
# Directory with images for external OCR fallback.
OCR_IMAGES_DIR=data/cropped
# Delay between page fetches from Telegram (seconds).
TELEGRAM_PAGE_SLEEP=1.0
# Delay between image downloads (seconds).
TELEGRAM_DOWNLOAD_SLEEP=0.3
# Remove *_thumb.* files when full image exists.
TELEGRAM_SKIP_THUMBS=1
# Base path for fetch logs; daily date will be appended.
TELEGRAM_LOG_FILE=data/telegram_fetch.log
# Manifest with Telegram metadata per file.
TELEGRAM_MANIFEST_FILE=data/telegram_manifest.json
# Base path for OCR logs; daily date will be appended.
OCR_LOG_FILE=data/ocr.log
# Base path for EasyOCR logs; daily date will be appended.
EASYOCR_LOG_FILE=data/easyocr.log
# EasyOCR languages (comma- or plus-separated).
EASYOCR_LANGS=ru,en
# Use GPU for EasyOCR (0/1).
EASYOCR_GPU=0
# Parallel EasyOCR workers.
EASYOCR_WORKERS=2
# Base path for preprocess logs; daily date will be appended.
PREPROCESS_LOG_FILE=data/preprocess.log
# Manifest for preprocess status.
PREPROCESS_MANIFEST_FILE=data/preprocess_manifest.json
# Directory for non-question images.
PREPROCESS_NOT_A_QUESTION_DIR=data/not_question
# Directory for cropped images used for OCR.
PREPROCESS_CROPPED_DIR=data/cropped
# Base path for Mistral moderation logs; daily date will be appended.
MISTRAL_LOG_FILE=data/mistral.log
# Base path for OCR.space logs; daily date will be appended.
OCRSPACE_LOG_FILE=data/ocrspace.log
# Base path for invalid Mistral batch responses; daily date will be appended.
MISTRAL_INVALID_LOG_FILE=data/mistral_invalid.log
# SQLite database path for validated phrases.
SQLITE_DB_PATH=data/questions.sqlite
# Base path for SQLite export logs; daily date will be appended.
SQLITE_LOG_FILE=data/sqlite_export.log
# Daily index JSON file path.
DAILY_INDEX_FILE=data/daily_index.json
# Base path for daily pipeline logs; daily date will be appended.
DAILY_LOG_FILE=data/telegram_daily.log
```

## Экспорт валидированных фраз в SQLite

Скрипт проходит по JSON и сохраняет фразы в SQLite в две таблицы:
`phrases_validated` (где `is_correct=true`) и `phrases_unvalidated`
(остальные записи). Файлы `*_ocr_failed_*`, `*_ocr_variants_*`
и `*_mistral_incorrect_*` игнорируются. По умолчанию строки, которые уже
есть в целевой таблице, не перезаписываются (можно отключить через
`--no-skip-existing`).

Пример запуска:

```
python3 scripts/export_validated_to_sqlite.py --input-dir data
```

Либо явно указать файлы:

```
python3 scripts/export_validated_to_sqlite.py --input-files data/questions_2025-12-22.json data/questions_2025-12-24.json
```

Очистить обе таблицы перед экспортом:

```
python3 scripts/export_validated_to_sqlite.py --truncate --input-dir data
```

Пример запуска:

```
python3 scripts/daily_pipeline.py --channel vopros_dna --out-dir data/photos
```

## Опционально: единый запуск

Если нужен один скрипт для всех шагов, можно использовать `scripts/daily_pipeline.py`.
Он объединяет скачивание, OCR, модерацию и финальный экспорт в SQLite,
но рекомендуется разделённый запуск.
