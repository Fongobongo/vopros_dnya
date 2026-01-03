import argparse
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
QUALITY_KEY = "is_correct"
DEFAULT_DB_PATH = "data/questions.sqlite"
VALIDATED_TABLE = "phrases_validated"
UNVALIDATED_TABLE = "phrases_unvalidated"
SKIP_MARKERS = ("_ocr_failed_", "_ocr_variants_", "_mistral_incorrect_")
DATE_RE = re.compile(r"\\d{4}-\\d{2}-\\d{2}")
FILENAME_DATE_RE = re.compile(r"@(?P<date>\\d{2}-\\d{2}-\\d{4})_")


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
    date_match = re.search(r"\\d{4}-\\d{2}-\\d{2}", base_path.name)
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
        date_match = re.search(r"\\d{4}-\\d{2}-\\d{2}", base_path.name)
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
    date_re = re.compile(r"\\d{4}-\\d{2}-\\d{2}")
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


def _load_records(path: Path) -> list[dict[str, Any]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    records = []
    for item in data:
        if isinstance(item, dict):
            records.append(item)
    return records


def _parse_date_from_filename(name: str) -> str | None:
    match = FILENAME_DATE_RE.search(name)
    if not match:
        return None
    raw = match.group("date")
    try:
        dt = datetime.strptime(raw, "%d-%m-%Y")
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d")


def _date_from_record(record: dict[str, Any], fallback: str | None) -> str | None:
    for key in ("tg_datetime_utc", "datetime"):
        value = record.get(key)
        if isinstance(value, str):
            if " " in value:
                return value.split(" ")[0]
            if "T" in value:
                return value.split("T")[0]
    filename = record.get("filename")
    if isinstance(filename, str):
        from_name = _parse_date_from_filename(filename)
        if from_name:
            return from_name
    return fallback


def _bool_to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    return None


def _init_db(conn: sqlite3.Connection) -> None:
    for table in (VALIDATED_TABLE, UNVALIDATED_TABLE):
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                number INTEGER,
                datetime TEXT,
                filename TEXT UNIQUE,
                text TEXT,
                tg_message_id INTEGER,
                tg_datetime_utc TEXT,
                is_correct INTEGER,
                llm_validated INTEGER,
                human_validated INTEGER,
                is_sexual INTEGER,
                is_profanity INTEGER,
                is_politics INTEGER,
                is_insults INTEGER,
                is_threats INTEGER,
                is_harassment INTEGER,
                is_twitch_banned INTEGER,
                source_json TEXT,
                imported_at_utc TEXT
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_date ON {table}(date)"
        )
    conn.commit()


def _truncate_tables(conn: sqlite3.Connection) -> None:
    for table in (VALIDATED_TABLE, UNVALIDATED_TABLE):
        conn.execute(f"DELETE FROM {table}")
    try:
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name IN (?, ?)",
            (VALIDATED_TABLE, UNVALIDATED_TABLE),
        )
    except sqlite3.OperationalError:
        pass
    conn.commit()


def _load_paths(args: argparse.Namespace) -> list[Path]:
    files = [Path(path) for path in args.json_files]
    if args.input_dir:
        input_dir = Path(args.input_dir)
        files.extend(sorted(input_dir.glob(args.pattern)))
    return [path for path in files if path.exists()]


def _fetch_existing_filenames(
    conn: sqlite3.Connection,
    table: str,
    filenames: list[str],
) -> set[str]:
    if not filenames:
        return set()
    found: set[str] = set()
    chunk_size = 900
    for i in range(0, len(filenames), chunk_size):
        chunk = filenames[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT filename FROM {table} WHERE filename IN ({placeholders})"
        rows = conn.execute(query, chunk).fetchall()
        found.update(row[0] for row in rows)
    return found


def _delete_filenames(conn: sqlite3.Connection, table: str, filenames: list[str]) -> int:
    if not filenames:
        return 0
    removed = 0
    chunk_size = 900
    for i in range(0, len(filenames), chunk_size):
        chunk = filenames[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        query = f"DELETE FROM {table} WHERE filename IN ({placeholders})"
        cur = conn.execute(query, chunk)
        removed += cur.rowcount or 0
    return removed


def _upsert_rows(conn: sqlite3.Connection, table: str, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    conn.executemany(
        f"""
        INSERT INTO {table} (
            date,
            number,
            datetime,
            filename,
            text,
            tg_message_id,
            tg_datetime_utc,
            is_correct,
            llm_validated,
            human_validated,
            is_sexual,
            is_profanity,
            is_politics,
            is_insults,
            is_threats,
            is_harassment,
            is_twitch_banned,
            source_json,
            imported_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            date=excluded.date,
            number=excluded.number,
            datetime=excluded.datetime,
            text=excluded.text,
            tg_message_id=excluded.tg_message_id,
            tg_datetime_utc=excluded.tg_datetime_utc,
            is_correct=excluded.is_correct,
            llm_validated=excluded.llm_validated,
            human_validated=excluded.human_validated,
            is_sexual=excluded.is_sexual,
            is_profanity=excluded.is_profanity,
            is_politics=excluded.is_politics,
            is_insults=excluded.is_insults,
            is_threats=excluded.is_threats,
            is_harassment=excluded.is_harassment,
            is_twitch_banned=excluded.is_twitch_banned,
            source_json=excluded.source_json,
            imported_at_utc=excluded.imported_at_utc
        """,
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export validated phrases from JSON files into SQLite.",
    )
    parser.add_argument("json_files", nargs="*", help="JSON files to process")
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory to scan for JSON files (default: data)",
    )
    parser.add_argument(
        "--pattern",
        default="questions_*.json",
        help="Filename pattern when using --input-dir (default: questions_*.json)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite DB path (default: env SQLITE_DB_PATH or data/questions.sqlite)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/sqlite_export.log)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip files where all filenames already exist in SQLite",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate all SQLite tables before export",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    log_file = args.log_file or os.getenv("SQLITE_LOG_FILE") or "data/sqlite_export.log"
    _setup_logging(Path(log_file))

    db_path = args.db_path or os.getenv("SQLITE_DB_PATH") or DEFAULT_DB_PATH
    db_path = str(Path(db_path))

    conn = sqlite3.connect(db_path)
    try:
        _init_db(conn)
        if args.truncate:
            _truncate_tables(conn)
            LOGGER.info("Truncated SQLite tables")

        files = _load_paths(args)
        if not files:
            LOGGER.warning("No JSON files to process.")
            return 0 if args.truncate else 1
        total_inserted_valid = 0
        total_inserted_invalid = 0
        total_skipped = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for path in files:
            if any(marker in path.stem for marker in SKIP_MARKERS):
                LOGGER.info("Skipping auxiliary file %s", path)
                continue
            records = _load_records(path)
            if records is None:
                LOGGER.warning("Invalid JSON format in %s", path)
                total_skipped += 1
                continue
            filenames = [
                record["filename"]
                for record in records
                if isinstance(record.get("filename"), str)
            ]
            if filenames:
                existing_valid = _fetch_existing_filenames(conn, VALIDATED_TABLE, filenames)
                existing_invalid = _fetch_existing_filenames(conn, UNVALIDATED_TABLE, filenames)
            else:
                existing_valid = set()
                existing_invalid = set()
            match = DATE_RE.search(path.name)
            file_date = match.group(0) if match else None

            valid_rows = []
            invalid_rows = []
            move_to_valid = []
            move_to_invalid = []
            for record in records:
                filename = record.get("filename")
                if not isinstance(filename, str):
                    continue
                date_value = _date_from_record(record, file_date)
                row = (
                    date_value,
                    record.get("number"),
                    record.get("datetime"),
                    filename,
                    record.get("text"),
                    record.get("tg_message_id"),
                    record.get("tg_datetime_utc"),
                    _bool_to_int(record.get("is_correct")),
                    _bool_to_int(record.get("llm_validated")),
                    _bool_to_int(record.get("human_validated")),
                    _bool_to_int(record.get("is_sexual")),
                    _bool_to_int(record.get("is_profanity")),
                    _bool_to_int(record.get("is_politics")),
                    _bool_to_int(record.get("is_insults")),
                    _bool_to_int(record.get("is_threats")),
                    _bool_to_int(record.get("is_harassment")),
                    _bool_to_int(record.get("is_twitch_banned")),
                    str(path),
                    now,
                )
                if record.get(QUALITY_KEY) is True:
                    if (
                        not args.no_skip_existing
                        and filename in existing_valid
                        and filename not in existing_invalid
                    ):
                        continue
                    if filename in existing_invalid:
                        move_to_valid.append(filename)
                    valid_rows.append(row)
                else:
                    if (
                        not args.no_skip_existing
                        and filename in existing_invalid
                        and filename not in existing_valid
                    ):
                        continue
                    if filename in existing_valid:
                        move_to_invalid.append(filename)
                    invalid_rows.append(row)
            if not valid_rows and not invalid_rows:
                total_skipped += 1
                LOGGER.info("Skipping %s (no new rows to export)", path)
                continue

            if move_to_valid:
                _delete_filenames(conn, UNVALIDATED_TABLE, move_to_valid)
            if move_to_invalid:
                _delete_filenames(conn, VALIDATED_TABLE, move_to_invalid)

            _upsert_rows(conn, VALIDATED_TABLE, valid_rows)
            _upsert_rows(conn, UNVALIDATED_TABLE, invalid_rows)
            conn.commit()

            if valid_rows:
                total_inserted_valid += len(valid_rows)
            if invalid_rows:
                total_inserted_invalid += len(invalid_rows)
            LOGGER.info(
                "Exported valid=%s invalid=%s from %s",
                len(valid_rows),
                len(invalid_rows),
                path,
            )

        LOGGER.info(
            "SQLite export finished: valid=%s invalid=%s skipped=%s db=%s",
            total_inserted_valid,
            total_inserted_invalid,
            total_skipped,
            db_path,
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
