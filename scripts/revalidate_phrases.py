import argparse
import json
import logging
import os
import re
import subprocess
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
SKIP_MARKERS = ("_ocr_failed_", "_ocr_variants_", "_mistral_incorrect_")
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


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_flag_defaults(record: dict[str, Any]) -> bool:
    changed = False
    for flag in FLAGS:
        if flag not in record:
            record[flag] = None
            changed = True
    return changed


def _set_default_if_missing(record: dict[str, Any], key: str) -> bool:
    if key not in record:
        record[key] = None
        return True
    return False


def _export_to_sqlite(env_file: Path, db_path: str | None, files: list[Path]) -> int:
    if not files:
        return 0
    script_path = ROOT_DIR / "scripts" / "export_validated_to_sqlite.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--input-files",
        *[str(path) for path in files],
        "--no-skip-existing",
        "--env-file",
        str(env_file),
    ]
    if db_path:
        cmd.extend(["--db-path", db_path])
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run phrase validation and Mistral restore, update moderation flags, "
            "and sync results to SQLite."
        ),
    )
    parser.add_argument("json_files", nargs="*", help="JSON files to process")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory to scan for JSON files",
    )
    parser.add_argument(
        "--pattern",
        default="questions_*.json",
        help="Filename pattern when using --input-dir (default: questions_*.json)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/revalidate.log)",
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
        "--invalid-log-file",
        default=None,
        help="Invalid response log path (default: env MISTRAL_INVALID_LOG_FILE)",
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
        "--no-export-sqlite",
        action="store_true",
        help="Do not export updates to SQLite",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite DB path (default: env SQLITE_DB_PATH)",
    )
    args = parser.parse_args()

    env_file = Path(args.env_file)
    _load_env_file(env_file)
    log_file = args.log_file or os.getenv("REVALIDATE_LOG_FILE") or "data/revalidate.log"
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
    invalid_log_path = Path(
        args.invalid_log_file or os.getenv("MISTRAL_INVALID_LOG_FILE", "data/mistral_invalid.log")
    )

    files = [Path(p) for p in args.json_files]
    if args.input_dir:
        input_dir = Path(args.input_dir)
        files.extend(sorted(input_dir.glob(args.pattern)))
    files = [path for path in files if path.exists()]
    if not files:
        LOGGER.warning("No JSON files to process.")
        return 1

    updated_files: list[Path] = []
    for path in files:
        if any(marker in path.stem for marker in SKIP_MARKERS):
            LOGGER.info("Skipping auxiliary file %s", path)
            continue
        records = _load_records(path)
        if records is None:
            LOGGER.warning("Invalid JSON format in %s", path)
            continue

        file_changed = False
        restored_count = 0
        corrected_count = 0
        to_moderate: list[dict[str, Any]] = []

        for record in records:
            if not isinstance(record, dict):
                continue
            if _update_flag_defaults(record):
                file_changed = True
            if _set_default_if_missing(record, "llm_validated"):
                file_changed = True
            if _set_default_if_missing(record, "human_validated"):
                file_changed = True

            text_before = _normalize_text_value(record.get("text"))
            tesseract_text = _normalize_text_value(record.get("tesseract_text"))
            easyocr_text = _normalize_text_value(record.get("easyocr_text"))
            ocrspace_text = _normalize_text_value(record.get("ocrspace_text"))
            mistral_text = _normalize_text_value(record.get("mistral_text"))

            if record.get("text") != text_before:
                record["text"] = text_before
                file_changed = True
            if record.get("tesseract_text") != tesseract_text:
                record["tesseract_text"] = tesseract_text
                file_changed = True
            if record.get("easyocr_text") != easyocr_text:
                record["easyocr_text"] = easyocr_text
                file_changed = True
            if record.get("ocrspace_text") != ocrspace_text:
                record["ocrspace_text"] = ocrspace_text
                file_changed = True
            if record.get("mistral_text") != mistral_text:
                record["mistral_text"] = mistral_text
                file_changed = True

            was_correct = record.get("is_correct") is True
            new_text = text_before
            new_is_correct = True if (was_correct and new_text) else None

            if not was_correct:
                consensus = _consensus_text(
                    [tesseract_text, easyocr_text, ocrspace_text, mistral_text]
                )
                if consensus:
                    new_text = consensus
                    new_is_correct = True
                elif needs_restore and api_key and (tesseract_text or easyocr_text or ocrspace_text):
                    try:
                        restored = _mistral_restore_text(
                            tesseract_text or "",
                            easyocr_text or "",
                            ocrspace_text or "",
                            api_key,
                            model,
                            mistral_timeout,
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Mistral restore failed for %s: %s", record.get("filename"), exc)
                        restored = None
                    if mistral_request_sleep > 0:
                        time.sleep(mistral_request_sleep)
                    if restored != mistral_text:
                        record["mistral_text"] = restored
                        mistral_text = restored
                        file_changed = True
                    if restored:
                        restored_count += 1
                    consensus = _consensus_text(
                        [tesseract_text, easyocr_text, ocrspace_text, mistral_text]
                    )
                    if consensus:
                        new_text = consensus
                        new_is_correct = True

            if new_text != text_before:
                record["text"] = new_text
                file_changed = True
                for flag in FLAGS:
                    if record.get(flag) is not None:
                        record[flag] = None
                if new_text:
                    to_moderate.append(record)
            elif new_text and any(record.get(flag) is None for flag in FLAGS):
                to_moderate.append(record)

            if record.get("is_correct") != new_is_correct:
                record["is_correct"] = new_is_correct
                file_changed = True
                if new_is_correct:
                    corrected_count += 1

        if needs_moderation and to_moderate:
            moderate_json._apply_mistral(
                to_moderate,
                api_key,
                model,
                invalid_log_path,
                batch_size,
                mistral_timeout,
                mistral_request_sleep,
            )
            file_changed = True

        if file_changed:
            _write_records(path, records)
            updated_files.append(path)
            LOGGER.info(
                "Updated %s (restored=%s, corrected=%s, moderated=%s)",
                path,
                restored_count,
                corrected_count,
                len(to_moderate) if needs_moderation else 0,
            )
        else:
            LOGGER.info("No changes for %s", path)

    if not args.no_export_sqlite and updated_files:
        db_path = args.db_path or os.getenv("SQLITE_DB_PATH")
        result = _export_to_sqlite(env_file, db_path, updated_files)
        if result != 0:
            LOGGER.warning("SQLite export failed with code %s", result)
            return result

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
