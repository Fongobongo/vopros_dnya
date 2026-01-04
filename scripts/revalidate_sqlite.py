import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import moderate_json


LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
DEFAULT_BATCH_SIZE = 10
DEFAULT_MISTRAL_TIMEOUT = 60.0
DEFAULT_MISTRAL_SLEEP = 0.0
MAX_TEXT_CHARS = 4000
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
VALIDATED_TABLE = "phrases_validated"
UNVALIDATED_TABLE = "phrases_unvalidated"
FLAGS = [
    "is_sexual",
    "is_profanity",
    "is_politics",
    "is_insults",
    "is_threats",
    "is_harassment",
    "is_twitch_banned",
    "is_ad",
    "is_racist",
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


def _resolve_float(value: float | None, env_key: str, default: float) -> float:
    if value is not None:
        return value
    raw = os.getenv(env_key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve_nonnegative_float(value: float | None, env_key: str, default: float) -> float:
    if value is not None:
        return max(0.0, value)
    raw = os.getenv(env_key)
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _resolve_int(value: int | None, env_key: str, default: int) -> int:
    if value is not None:
        return max(1, value)
    raw = os.getenv(env_key)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


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


def _extract_json(content: str) -> dict[str, Any] | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_text_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _consensus_text(values: list[str | None]) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
        if counts[value] >= 2:
            return value
    return None


def _mistral_restore_text(
    tesseract_text: str,
    easyocr_text: str,
    ocrspace_text: str,
    api_key: str,
    model: str,
    timeout: float,
) -> str | None:
    prompt = (
        "You are an OCR reconciliation assistant. "
        "Return ONLY JSON: {\"restored_text\": \"...\"}. "
        "Reconstruct the most likely original phrase using only information "
        "present in the three OCR variants. Do not add new words or phrases "
        "that do not appear in any input. You may fix obvious OCR mistakes "
        "(typos, missing/extra characters, spacing) and normalize spelling or "
        "letter case. Adjust punctuation only when it is clearly supported by "
        "the inputs; do not invent punctuation. "
        "If you cannot be confident, return an empty string."
    )
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "tesseract_text": tesseract_text[:MAX_TEXT_CHARS],
                        "easyocr_text": easyocr_text[:MAX_TEXT_CHARS],
                        "ocrspace_text": ocrspace_text[:MAX_TEXT_CHARS],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        MISTRAL_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        response_json = json.loads(resp.read().decode("utf-8"))
    content = response_json["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed:
        return None
    restored_text = parsed.get("restored_text")
    if not isinstance(restored_text, str):
        return None
    restored_text = restored_text.strip()
    return restored_text or None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _fetch_rows(conn: sqlite3.Connection, args: argparse.Namespace) -> list[sqlite3.Row]:
    conditions = []
    params: list[Any] = []
    if args.filenames:
        placeholders = ",".join("?" for _ in args.filenames)
        conditions.append(f"filename IN ({placeholders})")
        params.extend(args.filenames)
    if args.date_from:
        conditions.append("date >= ?")
        params.append(args.date_from)
    if args.date_to:
        conditions.append("date <= ?")
        params.append(args.date_to)

    query = f"SELECT * FROM {UNVALIDATED_TABLE}"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id"
    if args.limit is not None:
        query += " LIMIT ?"
        params.append(args.limit)
    if args.offset is not None:
        query += " OFFSET ?"
        params.append(args.offset)
    return conn.execute(query, params).fetchall()


def _normalize_flag_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    return None


def _normalize_llm_flag(value: Any) -> int | None:
    return 1 if value in (1, True) else None


def _prepare_values(record: dict[str, Any], columns: list[str], now: str) -> list[Any]:
    values = []
    for col in columns:
        if col == "imported_at_utc":
            values.append(now)
        else:
            values.append(record.get(col))
    return values


def _build_insert_sql(table: str, columns: list[str]) -> str:
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{col}=excluded.{col}" for col in columns)
    return (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(filename) DO UPDATE SET {updates}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate rows from phrases_unvalidated using consensus, "
            "Mistral restore, and moderation."
        ),
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
        help="Log file base path, supports {date} (default: data/revalidate_sqlite.log)",
    )
    parser.add_argument(
        "--mistral-api-key",
        default=None,
        help="Mistral API key (default: env MISTRAL_API_KEY)",
    )
    parser.add_argument(
        "--mistral-model",
        default=None,
        help="Mistral model name (default: env MISTRAL_MODEL or mistral-small-latest)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Request timeout in seconds (default: env MISTRAL_TIMEOUT or 60)",
    )
    parser.add_argument(
        "--mistral-request-sleep",
        type=float,
        default=None,
        help="Sleep between Mistral requests (default: env MISTRAL_REQUEST_SLEEP or 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Moderation batch size (default: env MISTRAL_BATCH_SIZE or 10)",
    )
    parser.add_argument(
        "--skip-mistral-restore",
        action="store_true",
        help="Skip Mistral restore requests",
    )
    parser.add_argument(
        "--skip-moderation",
        action="store_true",
        help="Skip moderation requests",
    )
    parser.add_argument(
        "--trust-text",
        action="store_true",
        help="Use existing text as authoritative when present",
    )
    parser.add_argument(
        "--filenames",
        nargs="*",
        default=None,
        help="Only revalidate these filenames",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="Only revalidate rows with date >= YYYY-MM-DD",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="Only revalidate rows with date <= YYYY-MM-DD",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Offset for row selection",
    )
    args = parser.parse_args()

    env_file = Path(args.env_file)
    _load_env_file(env_file)
    log_file = args.log_file or os.getenv("REVALIDATE_LOG_FILE") or "data/revalidate_sqlite.log"
    _setup_logging(Path(log_file))

    api_key = args.mistral_api_key or os.getenv("MISTRAL_API_KEY")
    needs_restore = not args.skip_mistral_restore
    needs_moderation = not args.skip_moderation
    if (needs_restore or needs_moderation) and not api_key:
        LOGGER.warning("MISTRAL_API_KEY not set; aborting.")
        return 2

    model = args.mistral_model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    mistral_timeout = _resolve_float(args.timeout, "MISTRAL_TIMEOUT", DEFAULT_MISTRAL_TIMEOUT)
    mistral_request_sleep = _resolve_nonnegative_float(
        args.mistral_request_sleep,
        "MISTRAL_REQUEST_SLEEP",
        DEFAULT_MISTRAL_SLEEP,
    )
    batch_size = _resolve_int(args.batch_size, "MISTRAL_BATCH_SIZE", DEFAULT_BATCH_SIZE)

    db_path = args.db_path or os.getenv("SQLITE_DB_PATH") or "data/questions.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_rows(conn, args)
        if not rows:
            LOGGER.info("No rows to process.")
            return 0

        validated_cols = [col for col in _table_columns(conn, VALIDATED_TABLE) if col != "id"]
        unvalidated_cols = [col for col in _table_columns(conn, UNVALIDATED_TABLE) if col != "id"]

        to_moderate: list[dict[str, Any]] = []
        to_update_unvalidated: list[dict[str, Any]] = []
        to_move_validated: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        restored_count = 0
        corrected_count = 0

        for row in rows:
            record = dict(row)
            record["text"] = _normalize_text_value(record.get("text"))
            record["tesseract_text"] = _normalize_text_value(record.get("tesseract_text"))
            record["easyocr_text"] = _normalize_text_value(record.get("easyocr_text"))
            record["ocrspace_text"] = _normalize_text_value(record.get("ocrspace_text"))
            record["mistral_text"] = _normalize_text_value(record.get("mistral_text"))
            record["llm_validated"] = _normalize_llm_flag(record.get("llm_validated"))
            record["human_validated"] = _normalize_llm_flag(record.get("human_validated"))

            consensus = None
            if args.trust_text and record.get("text"):
                consensus = record["text"]
            else:
                consensus = _consensus_text(
                    [
                        record["tesseract_text"],
                        record["easyocr_text"],
                        record["ocrspace_text"],
                        record["mistral_text"],
                    ]
                )
            if not consensus and needs_restore and api_key and not args.trust_text:
                if record["tesseract_text"] or record["easyocr_text"] or record["ocrspace_text"]:
                    try:
                        restored = _mistral_restore_text(
                            record["tesseract_text"] or "",
                            record["easyocr_text"] or "",
                            record["ocrspace_text"] or "",
                            api_key,
                            model,
                            mistral_timeout,
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Mistral restore failed for %s: %s",
                            record.get("filename"),
                            exc,
                        )
                        restored = None
                    if mistral_request_sleep > 0:
                        time.sleep(mistral_request_sleep)
                    if restored != record.get("mistral_text"):
                        record["mistral_text"] = restored
                    if restored:
                        restored_count += 1
                    consensus = _consensus_text(
                        [
                            record["tesseract_text"],
                            record["easyocr_text"],
                            record["ocrspace_text"],
                            record["mistral_text"],
                        ]
                    )

            if consensus:
                record["text"] = consensus
                record["is_correct"] = 1
                corrected_count += 1
                for flag in FLAGS:
                    record[flag] = None
                if needs_moderation:
                    to_moderate.append(record)
            else:
                record["text"] = None
                record["is_correct"] = None
                for flag in FLAGS:
                    record[flag] = None

            for flag in FLAGS:
                record[flag] = _normalize_flag_value(record.get(flag))

            if record["is_correct"] == 1 and record.get("text"):
                to_move_validated.append(record)
            else:
                to_update_unvalidated.append(record)

        if needs_moderation and to_moderate:
            moderate_json._apply_mistral(
                to_moderate,
                api_key,
                model,
                None,
                batch_size,
                mistral_timeout,
                mistral_request_sleep,
            )
            for record in to_moderate:
                for flag in FLAGS:
                    record[flag] = _normalize_flag_value(record.get(flag))

        update_sql = (
            f"UPDATE {UNVALIDATED_TABLE} SET "
            + ", ".join(f"{col}=?" for col in unvalidated_cols)
            + " WHERE id=?"
        )
        insert_sql = _build_insert_sql(VALIDATED_TABLE, validated_cols)

        with conn:
            if to_update_unvalidated:
                update_rows = [
                    _prepare_values(record, unvalidated_cols, now) + [record.get("id")]
                    for record in to_update_unvalidated
                ]
                conn.executemany(update_sql, update_rows)
            if to_move_validated:
                insert_rows = [
                    _prepare_values(record, validated_cols, now) for record in to_move_validated
                ]
                conn.executemany(insert_sql, insert_rows)
                conn.executemany(
                    f"DELETE FROM {UNVALIDATED_TABLE} WHERE id=?",
                    [(record.get("id"),) for record in to_move_validated],
                )

        LOGGER.info(
            "Processed %s rows (restored=%s, corrected=%s, moved=%s)",
            len(rows),
            restored_count,
            corrected_count,
            len(to_move_validated),
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
