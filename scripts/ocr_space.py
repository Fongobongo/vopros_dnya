import argparse
import io
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from scripts.retry_utils import load_retry_config, run_with_retry
LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
DEFAULT_TIMEOUT = 60.0
DEFAULT_REQUEST_SLEEP = 0.0
DEFAULT_ENGINE = 2
DEFAULT_LANGUAGE = "rus"
DEFAULT_API_URL = "https://api.ocr.space/parse/image"
DEFAULT_CROP = "0.08,0.18,0.08,0.20"
USER_AGENT = "Mozilla/5.0 (compatible; OCRFetcher/1.0; +https://t.me)"
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
        return max(1.0, value)
    raw = os.getenv(env_key)
    if not raw:
        return default
    try:
        return max(1.0, float(raw))
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


def _default_flags() -> dict[str, Any]:
    return {flag: None for flag in FLAGS}


def _list_images(input_dir: Path) -> list[dict[str, Any]]:
    full_images: dict[str, Path] = {}
    thumb_images: dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if path.name.endswith("_thumb.jpg"):
            base = path.name.replace("_thumb.jpg", ".jpg")
            thumb_images[base] = path
        else:
            full_images[path.name] = path

    items = []
    for name, full_path in full_images.items():
        chosen = thumb_images.get(name, full_path)
        items.append({"filename": name, "path": chosen})
    return items


def _load_existing_filenames(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, list):
        return set()
    filenames = set()
    for item in data:
        if isinstance(item, dict) and item.get("filename"):
            filenames.add(item["filename"])
    return filenames


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


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_index = {date: index[date] for date in sorted(index)}
    path.write_text(json.dumps(sorted_index, ensure_ascii=False, indent=2), encoding="utf-8")


def _date_from_value(value: str | None, fallback: datetime) -> str:
    if isinstance(value, str):
        if " " in value:
            return value.split(" ")[0]
        if "T" in value:
            return value.split("T")[0]
    return fallback.strftime("%Y-%m-%d")


def _ocr_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    extracted = 0
    for record in records:
        text = record.get("text") or ""
        if isinstance(text, str) and text.strip():
            extracted += 1
    total = len(records)
    failed = total - extracted
    return {"total": total, "extracted": extracted, "failed": failed}


def _parse_crop(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("Invalid crop format. Use: left,top,right,bottom")
    for part in parts:
        if part < 0 or part >= 1:
            raise ValueError("Crop values must be in [0, 1)")
    return tuple(parts)  # type: ignore[return-value]


def _crop_center(image: Image.Image, crop: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left_pct, top_pct, right_pct, bottom_pct = crop
    left = int(width * left_pct)
    top = int(height * top_pct)
    right = int(width * (1.0 - right_pct))
    bottom = int(height * (1.0 - bottom_pct))
    return image.crop((left, top, right, bottom))


def _image_to_jpeg_bytes(path: Path, crop: tuple[float, float, float, float]) -> bytes:
    image = Image.open(path).convert("RGB")
    image = _crop_center(image, crop)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _encode_multipart_form(
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"))
        lines.append(b"")
        lines.append(str(value).encode("utf-8"))
    for field_name, filename, data, content_type in files:
        lines.append(f"--{boundary}".encode("utf-8"))
        disposition = f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'
        lines.append(disposition.encode("utf-8"))
        lines.append(f"Content-Type: {content_type}".encode("utf-8"))
        lines.append(b"")
        lines.append(data)
    lines.append(f"--{boundary}--".encode("utf-8"))
    lines.append(b"")
    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _ocr_space_text(
    image_bytes: bytes,
    api_key: str,
    api_url: str,
    language: str,
    engine: int,
    timeout: float,
) -> str:
    fields = {
        "apikey": api_key,
        "language": language,
        "isOverlayRequired": "false",
        "OCREngine": str(engine),
    }
    files = [("file", "image.jpg", image_bytes, "image/jpeg")]
    body, content_type = _encode_multipart_form(fields, files)
    req = Request(
        api_url,
        data=body,
        headers={
            "Content-Type": content_type,
            "User-Agent": USER_AGENT,
        },
    )
    try:
        retry_config = load_retry_config("OCRSPACE", "REQUEST")

        def _do_request() -> dict[str, Any]:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _on_retry(attempt: int, total: int, delay: float, exc: Exception) -> None:
            LOGGER.warning(
                "OCR.space request failed (attempt %s/%s): %s; retrying in %.1fs",
                attempt,
                total,
                exc,
                delay,
            )

        response_json = run_with_retry(_do_request, retry_config, _on_retry)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        LOGGER.warning("OCR.space HTTP %s: %s", exc.code, detail.strip())
        return ""
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("OCR.space request failed: %s", exc)
        return ""
    if response_json.get("IsErroredOnProcessing"):
        message = response_json.get("ErrorMessage") or response_json.get("ErrorDetails")
        LOGGER.warning("OCR.space error: %s", message)
        return ""
    results = response_json.get("ParsedResults")
    if not isinstance(results, list) or not results:
        return ""
    parsed_text = results[0].get("ParsedText") if isinstance(results[0], dict) else ""
    if isinstance(parsed_text, str):
        return parsed_text.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OCR images with OCR.space and write per-date JSON files.",
    )
    parser.add_argument("input_dir", help="Directory with images")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON base path, supports {date} (default: data/questions_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OCR.space API key (default: env OCRSPACE_API_KEY)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="OCR.space API URL (default: env OCRSPACE_API_URL)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="OCR.space language (default: env OCRSPACE_LANGUAGE)",
    )
    parser.add_argument(
        "--engine",
        type=int,
        default=None,
        help="OCR.space engine (default: env OCRSPACE_OCR_ENGINE or 2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Request timeout in seconds (default: env OCRSPACE_TIMEOUT or 60)",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=None,
        help="Sleep between OCR.space requests (default: env OCRSPACE_REQUEST_SLEEP or 0)",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Crop as left,top,right,bottom (default: env EXTERNAL_OCR_CROP)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/ocrspace.log)",
    )
    parser.add_argument(
        "--manifest-file",
        default=None,
        help="Manifest JSON path (default: data/telegram_manifest.json)",
    )
    parser.add_argument(
        "--index-file",
        default=None,
        help="Index JSON path (default: data/daily_index.json)",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    log_file = args.log_file or os.getenv("OCRSPACE_LOG_FILE") or "data/ocrspace.log"
    _setup_logging(Path(log_file))

    api_key = args.api_key or os.getenv("OCRSPACE_API_KEY")
    if not api_key:
        LOGGER.warning("OCRSPACE_API_KEY not set; aborting.")
        return 2
    api_url = args.api_url or os.getenv("OCRSPACE_API_URL", DEFAULT_API_URL)
    language = args.language or os.getenv("OCRSPACE_LANGUAGE", DEFAULT_LANGUAGE)
    raw_engine = args.engine or os.getenv("OCRSPACE_OCR_ENGINE")
    try:
        engine = int(raw_engine) if raw_engine else DEFAULT_ENGINE
    except ValueError:
        engine = DEFAULT_ENGINE
    if engine not in {1, 2}:
        engine = DEFAULT_ENGINE
    timeout = _resolve_float(args.timeout, "OCRSPACE_TIMEOUT", DEFAULT_TIMEOUT)
    request_sleep = _resolve_nonnegative_float(
        args.request_sleep,
        "OCRSPACE_REQUEST_SLEEP",
        DEFAULT_REQUEST_SLEEP,
    )
    crop_raw = args.crop or os.getenv("EXTERNAL_OCR_CROP", DEFAULT_CROP)
    try:
        crop = _parse_crop(crop_raw)
    except ValueError as exc:
        LOGGER.warning("Invalid EXTERNAL_OCR_CROP: %s", exc)
        return 2

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        LOGGER.warning("Input directory not found: %s", input_dir)
        return 1

    now = datetime.now(timezone.utc)
    manifest_file = (
        args.manifest_file
        or os.getenv("TELEGRAM_MANIFEST_FILE")
        or "data/telegram_manifest.json"
    )
    index_file = args.index_file or os.getenv("DAILY_INDEX_FILE") or "data/daily_index.json"

    items = _list_images(input_dir)
    if not items:
        LOGGER.info("No images found in %s", input_dir)
        return 0

    manifest = _load_manifest(Path(manifest_file))
    for item in items:
        meta = manifest.get(item["filename"], {})
        item["tg_message_id"] = meta.get("message_id")
        item["tg_datetime_utc"] = meta.get("message_datetime_utc")
        item["date"] = _date_from_value(item["tg_datetime_utc"], now)

    existing_by_date: dict[str, set[str]] = {}
    todo = []
    for item in items:
        date_str = item["date"]
        if date_str not in existing_by_date:
            out_path = _build_output_path(args.out_json, date_str)
            existing_by_date[date_str] = _load_existing_filenames(out_path)
        if item["filename"] in existing_by_date[date_str]:
            continue
        todo.append(item)

    if not todo:
        LOGGER.info("No new images to OCR.")
        return 0

    records = []
    for item in todo:
        try:
            image_bytes = _image_to_jpeg_bytes(item["path"], crop)
            text = _ocr_space_text(image_bytes, api_key, api_url, language, engine, timeout)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("OCR.space failed for %s: %s", item["filename"], exc)
            text = ""
        if isinstance(text, str):
            text = text.strip() or None
        else:
            text = None
        if request_sleep > 0:
            time.sleep(request_sleep)
        record = {
            "number": None,
            "datetime": None,
            "filename": item["filename"],
            "text": text,
            "llm_validated": None,
            "human_validated": None,
            "is_correct": None,
            "tg_message_id": item.get("tg_message_id"),
            "tg_datetime_utc": item.get("tg_datetime_utc"),
        }
        record.update(_default_flags())
        records.append(record)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        date_str = _date_from_value(record.get("tg_datetime_utc"), now)
        grouped.setdefault(date_str, []).append(record)

    index_path = Path(index_file)
    index = _load_index(index_path)
    for date_str, items in grouped.items():
        out_path = _build_output_path(args.out_json, date_str)
        merged = _merge_existing(out_path, items)
        _write_json(out_path, merged)
        LOGGER.info("Wrote OCR output to %s", out_path)

        ocr = _ocr_stats(merged)
        entry = index.get(date_str, {})
        downloaded_count = entry.get("downloaded_count")
        if downloaded_count is None:
            downloaded_count = ocr["total"]
        download_failed_count = int(entry.get("download_failed_count", 0))
        downloaded_new_count = int(entry.get("downloaded_new_count", 0))
        download_failed_new_count = int(entry.get("download_failed_new_count", 0))
        entry.update(
            {
                "date": date_str,
                "json_path": str(out_path),
                "download_success": downloaded_count > 0 and download_failed_count == 0,
                "download_failed": download_failed_count > 0,
                "downloaded_count": int(downloaded_count),
                "downloaded_new_count": downloaded_new_count,
                "download_failed_count": download_failed_count,
                "download_failed_new_count": download_failed_new_count,
                "ocr_success": ocr["failed"] == 0 and ocr["total"] > 0,
                "ocr_failed": ocr["failed"] > 0,
                "ocr_failed_count": ocr["failed"],
                "ocr_extracted_phrases": ocr["extracted"],
            }
        )
        index[date_str] = entry

    if index:
        _write_index(index_path, index)
        LOGGER.info("Updated index %s", index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
