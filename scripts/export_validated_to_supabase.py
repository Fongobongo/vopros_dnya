import argparse
import logging
import os
import re
import sqlite3
import time
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supabase import create_client


LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FLAGS = [
    "is_sexual",
    "is_profanity",
    "is_politics",
    "is_religion",
    "is_insults",
    "is_threats",
    "is_harassment",
    "is_twitch_banned",
    "is_ad",
    "is_racist",
]
BOOL_FIELDS = {"is_correct", "llm_validated", "human_validated", *FLAGS}
SQLITE_COLUMNS = [
    "date",
    "number",
    "datetime",
    "filename",
    "text",
    "tesseract_text",
    "easyocr_text",
    "ocrspace_text",
    "mistral_text",
    "tg_message_id",
    "tg_datetime_utc",
    "is_correct",
    "llm_validated",
    "human_validated",
    "is_sexual",
    "is_profanity",
    "is_politics",
    "is_religion",
    "is_insults",
    "is_threats",
    "is_harassment",
    "is_twitch_banned",
    "is_ad",
    "is_racist",
    "source_json",
    "imported_at_utc",
]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _resolve_log_path(base_path: Path, now: datetime) -> Path:
    date_str = now.strftime("%Y-%m-%d")
    base_str = str(base_path)
    if "{date}" in base_str:
        return Path(base_str.format(date=date_str))
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", base_path.name)
    if date_match:
        replaced_name = base_path.name.replace(date_match.group(0), date_str, 1)
        resolved = base_path.with_name(replaced_name)
        return resolved if resolved.suffix else resolved.with_suffix(".log")
    suffix = base_path.suffix or ".log"
    return base_path.with_name(f"{base_path.stem}_{date_str}{suffix}")


def _cleanup_old_logs(base_path: Path, now: datetime) -> None:
    if LOG_RETENTION_DAYS <= 0:
        return
    base_str = str(base_path)
    if "{date}" in base_str:
        pattern_path = Path(base_str.format(date="*"))
    else:
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", base_path.name)
        if date_match:
            replaced_name = base_path.name.replace(date_match.group(0), "*", 1)
            pattern_path = base_path.with_name(replaced_name)
            if not pattern_path.suffix:
                pattern_path = pattern_path.with_suffix(".log")
        else:
            suffix = base_path.suffix or ".log"
            pattern_path = base_path.with_name(f"{base_path.stem}_*{suffix}")
    parent = pattern_path.parent
    if not parent.exists():
        return
    date_re = re.compile(r"\d{4}-\d{2}-\d{2}")
    for path in parent.glob(pattern_path.name):
        if not path.is_file():
            continue
        match = date_re.search(path.name)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(0), "%Y-%m-%d").date()
        except ValueError:
            continue
        if (now.date() - file_date).days > LOG_RETENTION_DAYS:
            try:
                path.unlink()
                LOGGER.info("Removed old log %s", path)
            except OSError as exc:
                LOGGER.warning("Failed to remove old log %s: %s", path, exc)


def _setup_logging(base_path: Path) -> Path:
    now = datetime.now(timezone.utc)
    log_path = _resolve_log_path(base_path, now)
    logger = logging.getLogger()
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s UTC %(levelname)s %(message)s")
        formatter.converter = time.gmtime

        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    _cleanup_old_logs(base_path, now)
    return log_path


def _parse_table(value: str) -> tuple[str, str]:
    if "." in value:
        schema, table = value.split(".", 1)
    else:
        schema, table = "public", value
    if not NAME_RE.match(schema) or not NAME_RE.match(table):
        raise ValueError(f"Invalid table name: {value}")
    return schema, table


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw in {"0", "1"}:
            return raw == "1"
    return None


def _prepare_row(row: sqlite3.Row) -> list[Any]:
    values = []
    for key in SQLITE_COLUMNS:
        value = row[key]
        if key in BOOL_FIELDS:
            value = _to_bool(value)
        values.append(value)
    return values


def _derive_supabase_url(db_url: str | None) -> str | None:
    if not db_url:
        return None
    parsed = urlparse(db_url)
    host = parsed.hostname or ""
    if host.startswith("db."):
        host = host[len("db.") :]
    if host:
        return f"https://{host}"
    return None


def _ddl(schema: str, phrases_table: str, usage_schema: str, usage_table: str) -> str:
    return (
        f"CREATE SCHEMA IF NOT EXISTS {schema};\n"
        f"CREATE TABLE IF NOT EXISTS {schema}.{phrases_table} (\n"
        "  id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,\n"
        "  date TEXT,\n"
        "  number INTEGER,\n"
        "  datetime TEXT,\n"
        "  filename TEXT UNIQUE,\n"
        "  text TEXT,\n"
        "  tesseract_text TEXT,\n"
        "  easyocr_text TEXT,\n"
        "  ocrspace_text TEXT,\n"
        "  mistral_text TEXT,\n"
        "  tg_message_id INTEGER,\n"
        "  tg_datetime_utc TEXT,\n"
        "  is_correct BOOLEAN,\n"
        "  llm_validated BOOLEAN,\n"
        "  human_validated BOOLEAN,\n"
        "  is_sexual BOOLEAN,\n"
        "  is_profanity BOOLEAN,\n"
        "  is_politics BOOLEAN,\n"
        "  is_religion BOOLEAN,\n"
        "  is_insults BOOLEAN,\n"
        "  is_threats BOOLEAN,\n"
        "  is_harassment BOOLEAN,\n"
        "  is_twitch_banned BOOLEAN,\n"
        "  is_ad BOOLEAN,\n"
        "  is_racist BOOLEAN,\n"
        "  source_json TEXT,\n"
        "  imported_at_utc TEXT\n"
        ");\n"
        f"CREATE SCHEMA IF NOT EXISTS {usage_schema};\n"
        f"CREATE TABLE IF NOT EXISTS {usage_schema}.{usage_table} (\n"
        f"  phrase_id INTEGER NOT NULL REFERENCES {schema}.{phrases_table}(id) ON DELETE CASCADE,\n"
        "  chat_id BIGINT NOT NULL,\n"
        "  joke_id BIGINT,\n"
        "  UNIQUE (phrase_id, chat_id, joke_id)\n"
        ");"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export validated phrases from SQLite into Supabase/Postgres.",
    )
    parser.add_argument(
        "--sqlite-path",
        default=None,
        help="SQLite DB path (default: env SQLITE_DB_PATH or data/questions.sqlite)",
    )
    parser.add_argument(
        "--pg-url",
        default=None,
        help="Postgres connection URL (default: env SUPABASE_DATABASE_URL)",
    )
    parser.add_argument(
        "--supabase-url",
        default=None,
        help="Supabase project URL (default: env SUPABASE_URL or derived from SUPABASE_DATABASE_URL)",
    )
    parser.add_argument(
        "--supabase-key",
        default=None,
        help="Supabase publishable API key (default: env SUPABASE_PUBLISH_API_KEY)",
    )
    parser.add_argument(
        "--table",
        default="public.phrases_validated",
        help="Target table (default: public.phrases_validated)",
    )
    parser.add_argument(
        "--usage-table",
        default="public.phrase_chat_usage",
        help="Usage table (default: public.phrase_chat_usage)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for inserts (default: 500)",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate target table before export",
    )
    parser.add_argument(
        "--print-ddl",
        action="store_true",
        help="Print SQL for table creation and exit",
    )
    parser.add_argument(
        "--skip-table-check",
        action="store_true",
        help="Skip table existence check",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/supabase_export.log)",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    log_file = args.log_file or os.getenv("SUPABASE_LOG_FILE") or "data/supabase_export.log"
    _setup_logging(Path(log_file))

    schema, phrases_table = _parse_table(args.table)
    usage_schema, usage_table = _parse_table(args.usage_table)
    if usage_schema != schema:
        LOGGER.warning("Usage table schema differs from phrases schema.")
    if args.print_ddl:
        print(_ddl(schema, phrases_table, usage_schema, usage_table))
        return 0

    sqlite_path = args.sqlite_path or os.getenv("SQLITE_DB_PATH") or "data/questions.sqlite"
    pg_url = args.pg_url or os.getenv("SUPABASE_DATABASE_URL")
    supabase_url = (
        args.supabase_url
        or os.getenv("SUPABASE_URL")
        or os.getenv("SUPABASE_PROJECT_URL")
        or _derive_supabase_url(pg_url)
    )
    supabase_key = (
        args.supabase_key
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_PUBLISH_API_KEY")
    )
    if not supabase_url or not supabase_key:
        LOGGER.warning("Supabase URL or publishable key missing; aborting.")
        return 2

    client = create_client(supabase_url, supabase_key)
    phrases_client = client.schema(schema).table(phrases_table)

    if not args.skip_table_check:
        try:
            phrases_client.select("filename").limit(1).execute()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Supabase table check failed: %s", exc)
            LOGGER.warning(
                "Create tables first using SQL:\n%s",
                _ddl(schema, phrases_table, usage_schema, usage_table),
            )
            return 2

    if args.truncate:
        LOGGER.warning("Truncate skipped: Supabase publishable key cannot run DDL.")

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    try:
        sqlite_cur = sqlite_conn.cursor()
        sqlite_cur.execute(f"SELECT {', '.join(SQLITE_COLUMNS)} FROM phrases_validated")
        total = 0
        while True:
            batch = sqlite_cur.fetchmany(args.batch_size)
            if not batch:
                break
            payload = []
            for row in batch:
                values = _prepare_row(row)
                payload.append(dict(zip(SQLITE_COLUMNS, values)))
            response = phrases_client.upsert(
                payload,
                on_conflict="filename",
            ).execute()
            if response is None:
                LOGGER.warning("Supabase upsert failed: empty response")
                return 2
            total += len(payload)
            LOGGER.info("Exported %s rows", total)
    finally:
        sqlite_conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
