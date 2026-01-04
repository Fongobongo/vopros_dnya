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

from scripts.ocr_images import build_records
from scripts.fetch_channel_images import fetch_channel_images


USER_AGENT = "Mozilla/5.0 (compatible; OCRFetcher/1.0; +https://t.me)"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MAX_TEXT_CHARS = 4000
LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
DEFAULT_BATCH_SIZE = 10
DEFAULT_MISTRAL_TIMEOUT = 60.0
DEFAULT_MISTRAL_SLEEP = 0.0
THUMB_RE = re.compile(r".+_thumb\\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
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


def _parse_crop(value: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Invalid crop format. Use: left,top,right,bottom")
    return tuple(parts)  # type: ignore[return-value]


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


def _resolve_bool(value: bool | None, env_key: str, default: bool) -> bool:
    if value is not None:
        return value
    raw = os.getenv(env_key)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
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


def _resolve_sleep(value: float | None, env_key: str, default: float) -> float:
    return _resolve_nonnegative_float(value, env_key, default)


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


def _log_invalid_mistral_response(content: str, base_path: Path) -> Path:
    now = datetime.now(timezone.utc)
    path = _resolve_log_path(base_path, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{timestamp} UTC\n")
        fh.write(content)
        if not content.endswith("\n"):
            fh.write("\n")
        fh.write("\n")
    return path


def _run_preprocess(images: list[dict[str, Any]], input_dir: Path, env_file: Path) -> int:
    if not images:
        return 0
    script_path = ROOT_DIR / "scripts" / "preprocess_questions.py"
    if not script_path.exists():
        LOGGER.warning("Preprocess script not found: %s", script_path)
        return 1
    file_args = [
        str(Path(item["path"]).resolve())
        for item in images
        if item.get("path")
    ]
    if not file_args:
        return 0
    cmd = [
        sys.executable,
        str(script_path),
        str(input_dir),
        "--env-file",
        str(env_file),
        "--files",
        *file_args,
    ]
    LOGGER.info("Running preprocess step")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.warning("Preprocess step failed with code %s", result.returncode)
    return result.returncode


def _remove_thumbs(target_dir: Path) -> int:
    if not target_dir.exists():
        return 0
    removed = 0
    for path in target_dir.rglob("*"):
        if not path.is_file():
            continue
        if THUMB_RE.match(path.name):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                LOGGER.warning("Failed to remove thumb %s: %s", path.name, exc)
    return removed


def _run_sqlite_export(env_file: Path) -> int:
    script_path = ROOT_DIR / "scripts" / "export_validated_to_sqlite.py"
    if not script_path.exists():
        LOGGER.warning("SQLite export script not found: %s", script_path)
        return 1
    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir",
        "data",
        "--pattern",
        "questions_*.json",
        "--env-file",
        str(env_file),
    ]
    LOGGER.info("Running SQLite export")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.warning("SQLite export failed with code %s", result.returncode)
    return result.returncode


def _build_output_path(out_json: str | None, date_str: str) -> Path:
    if out_json:
        path = Path(out_json)
    else:
        path = Path("data") / f"questions_{date_str}.json"
    if "{date}" in str(path):
        return Path(str(path).format(date=date_str))
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    if date_match:
        replaced = path.name.replace(date_match.group(0), date_str, 1)
        resolved = path.with_name(replaced)
        return resolved if resolved.suffix else resolved.with_suffix(".json")
    if date_str not in path.name:
        return path.with_name(f"{path.stem}_{date_str}{path.suffix}")
    return path


def _date_from_value(value: str | None, fallback: datetime) -> str:
    if isinstance(value, str):
        if " " in value:
            return value.split(" ")[0]
        if "T" in value:
            return value.split("T")[0]
    return fallback.strftime("%Y-%m-%d")


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _ocr_images(
    items: list[dict[str, Any]],
    crop: tuple[float, float, float, float],
    lang: str,
    psm: int,
    workers: int,
) -> list[dict[str, Any]]:
    if not items:
        return []
    ocr_items = []
    for item in items:
        path = Path(item["path"])
        ocr_items.append(
            {
                "filename": path.name,
                "path": path,
                "tg_message_id": item.get("message_id"),
                "tg_datetime_utc": item.get("message_datetime_utc"),
            }
        )
    return build_records(ocr_items, crop, lang, psm, workers)


def _group_by_date(records: list[dict[str, Any]], now: datetime) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dt_str = record.get("tg_datetime_utc") or record.get("datetime")
        date_str = _date_from_value(dt_str, now)
        grouped.setdefault(date_str, []).append(record)
    return grouped


def _merge_existing(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return records
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        existing = []
    if not isinstance(existing, list):
        existing = []
    seen = set()
    for item in existing:
        if isinstance(item, dict) and item.get("filename"):
            seen.add(item["filename"])
    merged = list(existing)
    for record in records:
        if record.get("filename") in seen:
            continue
        merged.append(record)
    return merged


def _download_stats_by_date(
    downloaded: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    now: datetime,
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for item in downloaded:
        date_str = _date_from_value(item.get("message_datetime_utc"), now)
        stats.setdefault(date_str, {"downloaded": 0, "failed": 0})
        stats[date_str]["downloaded"] += 1
    for item in failed:
        date_str = _date_from_value(item.get("message_datetime_utc"), now)
        stats.setdefault(date_str, {"downloaded": 0, "failed": 0})
        stats[date_str]["failed"] += 1
    return stats


def _ocr_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    extracted = 0
    for record in records:
        text = record.get("text") or ""
        if isinstance(text, str) and text.strip():
            extracted += 1
    total = len(records)
    failed = total - extracted
    return {"total": total, "extracted": extracted, "failed": failed}


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _write_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_index = {date: index[date] for date in sorted(index)}
    path.write_text(json.dumps(sorted_index, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_json(content: str) -> dict[str, Any] | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None


def _extract_json_any(content: str) -> Any | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start_list = content.find("[")
    end_list = content.rfind("]")
    if start_list != -1 and end_list != -1 and end_list > start_list:
        try:
            return json.loads(content[start_list : end_list + 1])
        except json.JSONDecodeError:
            pass
    start_obj = content.find("{")
    end_obj = content.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        try:
            return json.loads(content[start_obj : end_obj + 1])
        except json.JSONDecodeError:
            return None
    return None


def _normalize_flags(payload: dict[str, Any]) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    for flag in FLAGS:
        value = _normalize_bool(payload, flag)
        if isinstance(value, bool):
            normalized[flag] = value
    return normalized


def _normalize_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "false"}:
            return lower == "true"
    return None


def _normalize_mistral_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    flags = _normalize_flags(payload)
    if len(flags) != len(FLAGS):
        return None
    return flags


def _validate_batch_payload(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, list):
        return None
    valid_items: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "id" not in item:
            continue
        normalized = _normalize_mistral_payload(item)
        if not normalized:
            continue
        valid_items.append({"id": item["id"], **normalized})
    return valid_items if valid_items else None


def _mistral_request(
    text: str,
    api_key: str,
    model: str,
    timeout: float,
) -> dict[str, Any] | None:
    if not isinstance(text, str):
        text = ""
    trimmed = text.strip()
    if len(trimmed) > MAX_TEXT_CHARS:
        trimmed = trimmed[:MAX_TEXT_CHARS]
    system_prompt = (
        "You are a strict moderation assistant. Return only JSON with keys: "
        "is_sexual, is_profanity, is_politics, is_insults, is_threats, is_harassment, "
        "is_twitch_banned, is_ad, is_racist. Use true if a category is present or has clear signs. "
        "Use false only if you are confident it is absent. "
        "If text is empty or insufficient, return all censor flags true."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text:\n{trimmed}"},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        MISTRAL_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        response_json = json.loads(resp.read().decode("utf-8"))
    content = response_json["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed:
        return None
    return _normalize_mistral_payload(parsed)


def _mistral_batch_request(
    records: list[dict[str, Any]],
    api_key: str,
    model: str,
    invalid_log_path: Path | None = None,
    timeout: float = DEFAULT_MISTRAL_TIMEOUT,
) -> dict[str, dict[str, Any]] | None:
    items = []
    for record in records:
        text = record.get("text") or ""
        items.append(
            {
                "id": record.get("filename"),
                "text": text[:MAX_TEXT_CHARS],
            }
        )
    system_prompt = (
        "You are a strict moderation assistant. Return ONLY a JSON array. "
        "Each item must be an object with keys: id, is_sexual, is_profanity, "
        "is_politics, is_insults, is_threats, is_harassment, is_twitch_banned, is_ad, is_racist. "
        "Use true if the category is present or has clear signs. "
        "Use false only if you are confident it is absent. "
        "If text is empty or insufficient, return all censor flags true."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        MISTRAL_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        response_json = json.loads(resp.read().decode("utf-8"))
    content = response_json["choices"][0]["message"]["content"]
    parsed = _extract_json_any(content)
    valid = _validate_batch_payload(parsed)
    if not valid:
        if invalid_log_path:
            saved_to = _log_invalid_mistral_response(content, invalid_log_path)
            LOGGER.warning("Mistral batch response invalid; saved to %s", saved_to)
        else:
            LOGGER.warning("Mistral batch response invalid")
        return None
    result: dict[str, dict[str, Any]] = {}
    for item in valid:
        item_id = item.pop("id")
        result[str(item_id)] = item
    return result


def _chunked(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, size)
    return [records[i : i + size] for i in range(0, len(records), size)]


def _apply_mistral(
    records: list[dict[str, Any]],
    api_key: str,
    model: str,
    invalid_log_path: Path | None,
    batch_size: int,
    timeout: float,
    request_sleep: float,
) -> None:
    if not records:
        return
    chunks = _chunked(records, batch_size)
    for chunk in chunks:
        try:
            batch = _mistral_batch_request(
                chunk,
                api_key,
                model,
                invalid_log_path,
                timeout,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Mistral batch error: %s", exc)
            batch = None
        if not batch:
            LOGGER.warning("Mistral batch response invalid; falling back to per-item")
            for record in chunk:
                text = record.get("text") or ""
                try:
                    verdict = _mistral_request(text, api_key, model, timeout)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Mistral error for %s: %s", record.get("filename"), exc)
                    continue
                if request_sleep > 0:
                    time.sleep(request_sleep)
                if not verdict:
                    continue
                for key, value in verdict.items():
                    record[key] = value
            continue
        for record in chunk:
            item_id = str(record.get("filename"))
            verdict = batch.get(item_id)
            if not verdict:
                LOGGER.warning("Mistral missing result for %s", item_id)
                continue
            for key, value in verdict.items():
                record[key] = value
        if request_sleep > 0:
            time.sleep(request_sleep)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily pipeline: fetch images, OCR, and Mistral moderation.",
    )
    parser.add_argument("--channel", default="vopros_dna", help="Telegram channel name")
    parser.add_argument(
        "--out-dir",
        default="data/photos",
        help="Directory to save images",
    )
    parser.add_argument(
        "--state-file",
        default="data/telegram_state.json",
        help="JSON file storing last seen message id",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=24,
        help="Fallback window when state file is missing (default: 24)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Max pages to scan per run (default: 10)",
    )
    parser.add_argument(
        "--page-sleep",
        type=float,
        default=None,
        help="Sleep seconds between page fetches (default: 1.0)",
    )
    parser.add_argument(
        "--download-sleep",
        type=float,
        default=None,
        help="Sleep seconds between image downloads (default: 0.3)",
    )
    parser.add_argument(
        "--skip-thumbs",
        action="store_true",
        default=None,
        help="Remove *_thumb.* files when full image exists (default: env TELEGRAM_SKIP_THUMBS)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill history in chunks and resume with state",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Alias for --backfill",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download new images and exit",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON base path, supports {date} (default: data/questions_YYYY-MM-DD.json)",
    )
    parser.add_argument("--lang", default="rus+eng", help="Tesseract languages")
    parser.add_argument(
        "--crop",
        default="0.08,0.18,0.08,0.20",
        help="Crop percentages: left,top,right,bottom to remove",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=4,
        help="Tesseract page segmentation mode (default: 4)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel OCR workers (default: 2)",
    )
    parser.add_argument(
        "--mistral-model",
        default=None,
        help="Mistral model name (default: env MISTRAL_MODEL or mistral-small-latest)",
    )
    parser.add_argument(
        "--mistral-api-key",
        default=None,
        help="Mistral API key (default: env MISTRAL_API_KEY)",
    )
    parser.add_argument(
        "--mistral-request-sleep",
        type=float,
        default=None,
        help="Sleep between Mistral requests (default: env MISTRAL_REQUEST_SLEEP or 0)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/telegram_daily.log)",
    )
    parser.add_argument(
        "--index-file",
        default=None,
        help="Index JSON file path (default: data/daily_index.json)",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    log_file = args.log_file or os.getenv("DAILY_LOG_FILE") or "data/telegram_daily.log"
    _setup_logging(Path(log_file))
    api_key = args.mistral_api_key or os.getenv("MISTRAL_API_KEY")
    model = args.mistral_model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    page_sleep = _resolve_float(args.page_sleep, "TELEGRAM_PAGE_SLEEP", 1.0)
    download_sleep = _resolve_float(args.download_sleep, "TELEGRAM_DOWNLOAD_SLEEP", 0.3)
    skip_thumbs = _resolve_bool(args.skip_thumbs, "TELEGRAM_SKIP_THUMBS", False)
    invalid_log_path = Path(
        os.getenv("MISTRAL_INVALID_LOG_FILE", "data/mistral_invalid.log")
    )
    batch_size = _resolve_int(None, "MISTRAL_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    mistral_timeout = _resolve_float(None, "MISTRAL_TIMEOUT", DEFAULT_MISTRAL_TIMEOUT)
    mistral_request_sleep = _resolve_sleep(
        args.mistral_request_sleep,
        "MISTRAL_REQUEST_SLEEP",
        DEFAULT_MISTRAL_SLEEP,
    )

    now = datetime.now(timezone.utc)
    index_file = args.index_file or os.getenv("DAILY_INDEX_FILE") or "data/daily_index.json"

    result = fetch_channel_images(
        channel=args.channel,
        out_dir=Path(args.out_dir),
        state_file=Path(args.state_file),
        since_hours=args.since_hours,
        max_pages=args.max_pages,
        backfill=args.backfill or args.full_history,
        page_sleep=page_sleep,
        download_sleep=download_sleep,
        skip_thumbs=skip_thumbs,
    )
    downloaded = result["downloaded"]
    failed = result["failed"]
    LOGGER.info("Fetched %s new images", len(downloaded))

    if args.download_only:
        return 0

    removed_thumbs = _remove_thumbs(Path(args.out_dir))
    if removed_thumbs:
        LOGGER.info("Removed %s thumbnail files", removed_thumbs)
    preprocess_code = _run_preprocess(downloaded, Path(args.out_dir), Path(args.env_file))
    if preprocess_code != 0:
        LOGGER.warning("Preprocess step failed; continuing with remaining images.")
    cropped_dir = Path(os.getenv("PREPROCESS_CROPPED_DIR", "data/cropped"))
    cropped_items = []
    skipped_no_crop = 0
    for item in downloaded:
        name = Path(item["path"]).name
        cropped_path = cropped_dir / name
        if cropped_path.exists():
            updated = dict(item)
            updated["path"] = str(cropped_path)
            cropped_items.append(updated)
        else:
            skipped_no_crop += 1
    if skipped_no_crop:
        LOGGER.info("Skipped %s images without crop output", skipped_no_crop)
    if not cropped_items:
        LOGGER.info("No images left after preprocess step.")
        return 0

    crop = _parse_crop(args.crop)
    records = _ocr_images(cropped_items, crop, args.lang, args.psm, args.workers)
    if not records:
        return 0

    grouped = _group_by_date(records, now)
    download_stats = _download_stats_by_date(downloaded, failed, now)
    index_path = Path(index_file)
    index = _load_index(index_path)
    merged_by_path: dict[Path, list[dict[str, Any]]] = {}
    new_records_by_path: dict[Path, list[dict[str, Any]]] = {}
    for date_str, items in grouped.items():
        out_path = _build_output_path(args.out_json, date_str)
        merged = _merge_existing(out_path, items)
        merged_by_path[out_path] = merged
        new_records_by_path[out_path] = items
        _write_json(out_path, merged)
        LOGGER.info("Wrote OCR output to %s", out_path)
        ocr = _ocr_stats(merged)
        download_stat = download_stats.get(date_str, {"downloaded": 0, "failed": 0})
        download_failed_total = int(index.get(date_str, {}).get("download_failed_count", 0))
        download_failed_total += download_stat["failed"]
        download_success = download_failed_total == 0 and ocr["total"] > 0
        ocr_success = ocr["failed"] == 0 and ocr["total"] > 0
        LOGGER.info(
            "Date %s new downloads=%s new download failures=%s",
            date_str,
            download_stat["downloaded"],
            download_stat["failed"],
        )
        index[date_str] = {
            "date": date_str,
            "json_path": str(out_path),
            "download_success": download_success,
            "download_failed": download_failed_total > 0,
            "downloaded_count": ocr["total"],
            "downloaded_new_count": download_stat["downloaded"],
            "download_failed_count": download_failed_total,
            "download_failed_new_count": download_stat["failed"],
            "ocr_success": ocr_success,
            "ocr_failed": ocr["failed"] > 0,
            "ocr_failed_count": ocr["failed"],
            "ocr_extracted_phrases": ocr["extracted"],
        }
    if index:
        _write_index(index_path, index)
        LOGGER.info("Updated index %s", index_path)

    if not api_key:
        LOGGER.warning("MISTRAL_API_KEY not set; skipping moderation.")
        return 2

    for out_path, items in new_records_by_path.items():
        if not items:
            continue
        _apply_mistral(
            items,
            api_key,
            model,
            invalid_log_path,
            batch_size,
            mistral_timeout,
            mistral_request_sleep,
        )
        merged = merged_by_path.get(out_path, items)
        _write_json(out_path, merged)
        LOGGER.info("Updated moderation flags in %s", out_path)
    export_code = _run_sqlite_export(Path(args.env_file))
    return 0 if export_code == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
