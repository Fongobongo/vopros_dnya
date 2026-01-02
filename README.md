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
python scripts/extract_questions.py ./vopros_dna/photos --out data/questions.json
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
python3 scripts/fetch_channel_images.py --channel vopros_dna --out-dir vopros_dna/photos
```

Частичный бэктфил с продолжением в следующих запусках:

```
python3 scripts/fetch_channel_images.py --channel vopros_dna --out-dir vopros_dna/photos --backfill --max-pages 50
```

Ограничение скорости запросов:

```
python3 scripts/fetch_channel_images.py --channel vopros_dna --out-dir vopros_dna/photos --page-sleep 1.0 --download-sleep 0.3
```

Логи пишутся в консоль и файл с датой (UTC время в каждой строке),
например `data/telegram_fetch_YYYY-MM-DD.log`. Логи старше 30 дней удаляются
автоматически. Можно использовать `{date}` в пути. Базовый путь можно задать так:

```
python3 scripts/fetch_channel_images.py --channel vopros_dna --out-dir vopros_dna/photos --log-file data/telegram_fetch.log
```

Пример ежедневного запуска в полночь через cron:

```
0 0 * * * /opt/ocr/.venv/bin/python /opt/ocr/scripts/fetch_channel_images.py --channel vopros_dna --out-dir /opt/ocr/vopros_dna/photos --state-file /opt/ocr/data/telegram_state.json
```

Скрипт обновляет манифест `data/telegram_manifest.json` (метаданные постов)
и индекс `data/daily_index.json` (статистика загрузки).
Пути можно переопределить через `--manifest-file` и `--index-file`.

## Формат JSON и индекс

OCR скрипт создаёт отдельный JSON на каждую дату (UTC), а скрипт модерации
обновляет флаги цензуры через Mistral AI.
По умолчанию файлы создаются как `data/questions_YYYY-MM-DD.json`.
Внутри для каждой картинки есть `number`, `datetime`, `filename`, `text`,
`llm_validated`, `human_validated`, `is_correct`, `tg_message_id`, `tg_datetime_utc`
и поля цензуры `is_sexual`,
`is_profanity`, `is_politics`, `is_insults`, `is_threats`, `is_harassment`,
`is_twitch_banned` (изначально `true`). Время из Telegram пишется в UTC+0.
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

Пример запуска:

```
python3 scripts/ocr_images.py ./vopros_dna/photos
```

Переопределить путь можно через `--out-json` (поддерживает `{date}`).
Индекс можно переопределить через `--index-file` (или `DAILY_INDEX_FILE`).

Логи пишутся в консоль и файл с датой (UTC время в каждой строке),
например `data/ocr_YYYY-MM-DD.log`. Логи старше 30 дней удаляются автоматически.

## OCR через Mistral (vision)

Скрипт делает OCR через Mistral и пишет JSON по датам (UTC).

Пример запуска:

```
python3 scripts/mistral_ocr.py ./vopros_dna/photos
```

## Модерация JSON через Mistral

Скрипт обновляет флаги цензуры в указанных JSON файлах.
После первого прогона для записей с `is_correct=false` он делает OCR через Mistral,
пишет файл `*_ai_ocr.json`, снова прогоняет модерацию и:
- если `is_correct=false` — переносит запись в `*_ocr_failed_YYYY-MM-DD.json`
  (дата в UTC) и удаляет её из основного JSON;
- если `is_correct=true` — ставит `llm_validated=true` и обновляет запись в основном JSON.
Для OCR-фоллбэка нужен каталог изображений (`--images-dir` или `OCR_IMAGES_DIR`).

Пример запуска:

```
python3 scripts/moderate_json.py data/questions_2024-10-07.json --images-dir vopros_dna/photos
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
# Mistral OCR model (vision).
MISTRAL_OCR_MODEL=mistral-ocr-3
# Mistral OCR request timeout in seconds.
MISTRAL_OCR_TIMEOUT=60
# Directory with images for Mistral OCR fallback.
OCR_IMAGES_DIR=vopros_dna/photos
# Delay between page fetches from Telegram (seconds).
TELEGRAM_PAGE_SLEEP=1.0
# Delay between image downloads (seconds).
TELEGRAM_DOWNLOAD_SLEEP=0.3
# Base path for fetch logs; daily date will be appended.
TELEGRAM_LOG_FILE=data/telegram_fetch.log
# Manifest with Telegram metadata per file.
TELEGRAM_MANIFEST_FILE=data/telegram_manifest.json
# Base path for OCR logs; daily date will be appended.
OCR_LOG_FILE=data/ocr.log
# Base path for Mistral moderation logs; daily date will be appended.
MISTRAL_LOG_FILE=data/mistral.log
# Base path for Mistral OCR logs; daily date will be appended.
MISTRAL_OCR_LOG_FILE=data/mistral_ocr.log
# Base path for invalid Mistral batch responses; daily date will be appended.
MISTRAL_INVALID_LOG_FILE=data/mistral_invalid.log
# Daily index JSON file path.
DAILY_INDEX_FILE=data/daily_index.json
# Base path for daily pipeline logs; daily date will be appended.
DAILY_LOG_FILE=data/telegram_daily.log
```

Пример запуска:

```
python3 scripts/daily_pipeline.py --channel vopros_dna --out-dir vopros_dna/photos
```

## Опционально: единый запуск

Если нужен один скрипт для всех шагов, можно использовать `scripts/daily_pipeline.py`.
Он объединяет скачивание, OCR и модерацию, но рекомендуется разделённый запуск.
