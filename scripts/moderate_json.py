import argparse
import base64
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
DEFAULT_BATCH_SIZE = 10
DEFAULT_MISTRAL_TIMEOUT = 60.0
DEFAULT_OCR_TIMEOUT = 60.0
DEFAULT_OCR_MODEL = "mistral-ocr-3"
QUALITY_KEY = "is_correct"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MAX_TEXT_CHARS = 4000
FLAGS = [
    "is_sexual",
    "is_profanity",
    "is_politics",
    "is_insults",
    "is_threats",
    "is_harassment",
    "is_twitch_banned",
]
USER_AGENT = "Mozilla/5.0 (compatible; OCRFetcher/1.0; +https://t.me)"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


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


def _resolve_float(value: float | None, env_key: str, default: float) -> float:
    if value is not None:
        return max(1.0, value)
    raw = os.getenv(env_key)
    if not raw:
        return default
    try:
        return max(1.0, float(raw))
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
    is_correct = _normalize_bool(payload, QUALITY_KEY)
    if len(flags) != len(FLAGS) or not isinstance(is_correct, bool):
        return None
    return {**flags, QUALITY_KEY: is_correct}


def _default_flags() -> dict[str, bool]:
    return {flag: True for flag in FLAGS}


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
    trimmed = text.strip()
    if len(trimmed) > MAX_TEXT_CHARS:
        trimmed = trimmed[:MAX_TEXT_CHARS]
    system_prompt = (
        "You are a moderation assistant. Return only JSON with keys: is_sexual, is_profanity, "
        "is_politics, is_insults, is_threats, is_harassment, is_twitch_banned, is_correct. "
        "Use true if a category is present or has clear signs. Use false only if confident absent. "
        "Set is_correct=true only if the phrase is readable, meaningful, and has no obvious OCR "
        "garbage or stray symbols; otherwise false. If text is empty or insufficient, return all "
        "censor flags true and is_correct=false."
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
        "You are a moderation assistant. Return ONLY a JSON array. "
        "Each item must be an object with keys: id, is_sexual, is_profanity, "
        "is_politics, is_insults, is_threats, is_harassment, is_twitch_banned, is_correct. "
        "Use true if the category is present or has clear signs. "
        "Use false only if you are confident it is absent. "
        "Set is_correct=true only if the phrase is readable, meaningful, and has no obvious OCR "
        "garbage or stray symbols; otherwise false. "
        "If the text is empty or insufficient, return all censor flags true and is_correct=false."
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
                text = record.get("text", "")
                try:
                    verdict = _mistral_request(text, api_key, model, timeout)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Mistral error for %s: %s", record.get("filename"), exc)
                    continue
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


def _mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _mistral_ocr(
    path: Path,
    api_key: str,
    model: str,
    timeout: float,
) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = _mime_type(path)
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are an OCR assistant. Return only the extracted text.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all readable text."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
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
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        response_json = json.loads(resp.read().decode("utf-8"))
    content = response_json["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content.strip()
    return ""


def _load_json_records(path: Path) -> list[dict[str, Any]] | None:
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


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_by_filename(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    existing = _load_json_records(path) if path.exists() else []
    if existing is None:
        existing = []
    seen = set()
    for item in existing:
        filename = item.get("filename")
        if isinstance(filename, str):
            seen.add(filename)
    merged = list(existing)
    for record in records:
        filename = record.get("filename")
        if isinstance(filename, str) and filename in seen:
            continue
        merged.append(record)
    _write_json(path, merged)


def _build_ai_ocr_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_ai_ocr{path.suffix}")


def _build_ocr_failed_path(path: Path, date_str: str) -> Path:
    return path.with_name(f"{path.stem}_ocr_failed_{date_str}{path.suffix}")


def _build_image_index(images_dir: Path) -> dict[str, Path]:
    full_images: dict[str, Path] = {}
    thumb_images: dict[str, Path] = {}
    for path in images_dir.iterdir():
        if path.is_dir():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.name.endswith("_thumb.jpg"):
            base = path.name.replace("_thumb.jpg", ".jpg")
            thumb_images[base] = path
        else:
            full_images[path.name] = path
    resolved: dict[str, Path] = {}
    for name, full_path in full_images.items():
        resolved[name] = thumb_images.get(name, full_path)
    for name, thumb_path in thumb_images.items():
        resolved.setdefault(name, thumb_path)
    return resolved


def _ai_ocr_records(
    records: list[dict[str, Any]],
    images_dir: Path,
    api_key: str,
    model: str,
    timeout: float,
) -> list[dict[str, Any]]:
    image_index = _build_image_index(images_dir)
    results: list[dict[str, Any]] = []
    for record in records:
        filename = record.get("filename")
        if not isinstance(filename, str):
            continue
        image_path = image_index.get(filename)
        if not image_path:
            LOGGER.warning("Image not found for %s", filename)
            text = ""
        else:
            try:
                text = _mistral_ocr(image_path, api_key, model, timeout)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Mistral OCR failed for %s: %s", filename, exc)
                text = ""
        results.append(
            {
                "number": record.get("number"),
                "datetime": record.get("datetime"),
                "filename": filename,
                "text": text,
                "llm_validated": False,
                "human_validated": record.get("human_validated") is True,
                "is_correct": False,
                "tg_message_id": record.get("tg_message_id"),
                "tg_datetime_utc": record.get("tg_datetime_utc"),
                **_default_flags(),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Moderate OCR JSON files with Mistral.",
    )
    parser.add_argument("json_files", nargs="*", help="JSON files to update")
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
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: env MISTRAL_BATCH_SIZE or 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Request timeout in seconds (default: env MISTRAL_TIMEOUT or 60)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory with images for Mistral OCR fallback (default: env OCR_IMAGES_DIR)",
    )
    parser.add_argument(
        "--ocr-model",
        default=None,
        help="Mistral OCR model (default: env MISTRAL_OCR_MODEL or mistral-ocr-3)",
    )
    parser.add_argument(
        "--ocr-timeout",
        type=float,
        default=None,
        help="Mistral OCR timeout (default: env MISTRAL_OCR_TIMEOUT or 60)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/mistral.log)",
    )
    parser.add_argument(
        "--invalid-log-file",
        default=None,
        help="Invalid response log path (default: env MISTRAL_INVALID_LOG_FILE)",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    log_file = args.log_file or os.getenv("MISTRAL_LOG_FILE") or "data/mistral.log"
    _setup_logging(Path(log_file))

    api_key = args.mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        LOGGER.warning("MISTRAL_API_KEY not set; aborting.")
        return 2
    model = args.mistral_model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    batch_size = _resolve_int(args.batch_size, "MISTRAL_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    mistral_timeout = _resolve_float(args.timeout, "MISTRAL_TIMEOUT", DEFAULT_MISTRAL_TIMEOUT)
    invalid_log_path = Path(
        args.invalid_log_file or os.getenv("MISTRAL_INVALID_LOG_FILE", "data/mistral_invalid.log")
    )
    images_dir_value = args.images_dir or os.getenv("OCR_IMAGES_DIR")
    ocr_model = args.ocr_model or os.getenv("MISTRAL_OCR_MODEL", DEFAULT_OCR_MODEL)
    ocr_timeout = _resolve_float(args.ocr_timeout, "MISTRAL_OCR_TIMEOUT", DEFAULT_OCR_TIMEOUT)

    files = [Path(p) for p in args.json_files]
    if args.input_dir:
        input_dir = Path(args.input_dir)
        files.extend(sorted(input_dir.glob(args.pattern)))
    files = [path for path in files if path.exists()]
    if not files:
        LOGGER.warning("No JSON files to process.")
        return 1

    for path in files:
        if path.stem.endswith("_ai_ocr") or "_ocr_failed_" in path.stem:
            LOGGER.info("Skipping auxiliary file %s", path)
            continue
        records = _load_json_records(path)
        if records is None:
            LOGGER.warning("Invalid JSON format in %s", path)
            continue
        LOGGER.info("Moderating %s (%s records)", path, len(records))
        _apply_mistral(records, api_key, model, invalid_log_path, batch_size, mistral_timeout)
        incorrect = [record for record in records if record.get(QUALITY_KEY) is not True]
        if not incorrect:
            _write_json(path, records)
            LOGGER.info("Updated %s", path)
            continue
        if not images_dir_value:
            LOGGER.warning("OCR_IMAGES_DIR not set; skipping Mistral OCR fallback for %s", path)
            _write_json(path, records)
            continue
        images_dir = Path(images_dir_value)
        if not images_dir.exists():
            LOGGER.warning("Images dir not found: %s", images_dir)
            _write_json(path, records)
            continue

        LOGGER.info("Running Mistral OCR for %s records", len(incorrect))
        ai_records = _ai_ocr_records(incorrect, images_dir, api_key, ocr_model, ocr_timeout)
        ai_path = _build_ai_ocr_path(path)
        if ai_records:
            _write_json(ai_path, ai_records)
            LOGGER.info("Wrote AI OCR output to %s", ai_path)
            _apply_mistral(ai_records, api_key, model, invalid_log_path, batch_size, mistral_timeout)
            _write_json(ai_path, ai_records)

        corrected: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for record in ai_records:
            if record.get(QUALITY_KEY) is True:
                record["llm_validated"] = True
                corrected.append(record)
            else:
                failed.append(record)

        if corrected:
            corrected_by_filename = {
                item["filename"]: item
                for item in corrected
                if isinstance(item.get("filename"), str)
            }
            for record in records:
                filename = record.get("filename")
                if isinstance(filename, str) and filename in corrected_by_filename:
                    record.update(corrected_by_filename[filename])
        if failed:
            failed_filenames = {
                item.get("filename") for item in failed if isinstance(item.get("filename"), str)
            }
            records = [
                record
                for record in records
                if record.get("filename") not in failed_filenames
            ]
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            failed_path = _build_ocr_failed_path(path, today)
            _merge_by_filename(failed_path, failed)
            LOGGER.info("Wrote OCR failed output to %s", failed_path)

        _write_json(path, records)
        LOGGER.info("Updated %s (corrected: %s, failed: %s)", path, len(corrected), len(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
