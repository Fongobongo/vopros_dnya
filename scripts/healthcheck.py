import os
import sqlite3
import tempfile
from pathlib import Path


LOG_ENV_KEYS = [
    "TELEGRAM_LOG_FILE",
    "OCR_LOG_FILE",
    "EASYOCR_LOG_FILE",
    "PREPROCESS_LOG_FILE",
    "DAILY_LOG_FILE",
    "MISTRAL_LOG_FILE",
    "OCRSPACE_LOG_FILE",
    "MISTRAL_INVALID_LOG_FILE",
    "SQLITE_LOG_FILE",
]


def _ensure_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, prefix=".healthcheck_", delete=True) as tmp:
        tmp.write(b"ok")
        tmp.flush()


def _check_sqlite(db_path: Path) -> None:
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA quick_check")
    finally:
        conn.close()


def main() -> int:
    db_path = Path(os.getenv("SQLITE_DB_PATH", "data/questions.sqlite"))
    _ensure_writable(db_path.parent)
    _check_sqlite(db_path)

    for key in LOG_ENV_KEYS:
        value = os.getenv(key)
        if not value:
            continue
        _ensure_writable(Path(value).parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
