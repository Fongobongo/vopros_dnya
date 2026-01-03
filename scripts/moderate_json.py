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

LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
DEFAULT_BATCH_SIZE = 10
DEFAULT_MISTRAL_TIMEOUT = 60.0
DEFAULT_MISTRAL_SLEEP = 0.0
DEFAULT_OCRSPACE_TIMEOUT = 60.0
DEFAULT_OCRSPACE_SLEEP = 0.0
DEFAULT_OCRSPACE_ENGINE = 2
DEFAULT_OCRSPACE_LANGUAGE = "rus"
DEFAULT_OCRSPACE_URL = "https://api.ocr.space/parse/image"
DEFAULT_EXTERNAL_CROP = "0.08,0.18,0.08,0.20"
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
    if len(flags) != len(FLAGS):
        return None
    return flags


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


def _normalize_restore_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    restored_text = payload.get("restored_text")
    if not isinstance(restored_text, str):
        return None
    is_confident = _normalize_bool(payload, "is_confident")
    if is_confident is None:
        return None
    return {"restored_text": restored_text.strip(), "is_confident": is_confident}


def _validate_restore_payload(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, list):
        return None
    valid_items: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "id" not in item:
            continue
        normalized = _normalize_restore_payload(item)
        if not normalized:
            continue
        valid_items.append({"id": item["id"], **normalized})
    return valid_items if valid_items else None


def _mistral_restore_batch_request(
    records: list[dict[str, Any]],
    api_key: str,
    model: str,
    invalid_log_path: Path | None = None,
    timeout: float = DEFAULT_MISTRAL_TIMEOUT,
) -> dict[str, dict[str, Any]] | None:
    items = []
    for record in records:
        items.append(
            {
                "id": record.get("filename"),
                "tesseract_text": (record.get("tesseract_text") or "")[:MAX_TEXT_CHARS],
                "ocr_space_text": (record.get("ocr_space_text") or "")[:MAX_TEXT_CHARS],
            }
        )
    system_prompt = (
        "You are an OCR correction assistant. Return ONLY a JSON array. "
        "Each item must be an object with keys: id, restored_text, is_confident. "
        "Use the OCR variants to reconstruct the most likely original phrase. "
        "If you are not confident or all inputs are empty/garbled, return restored_text=\"\" "
        "and is_confident=false. Do not add commentary."
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
    valid = _validate_restore_payload(parsed)
    if not valid:
        if invalid_log_path:
            saved_to = _log_invalid_mistral_response(content, invalid_log_path)
            LOGGER.warning("Mistral restore response invalid; saved to %s", saved_to)
        else:
            LOGGER.warning("Mistral restore response invalid")
        return None
    result: dict[str, dict[str, Any]] = {}
    for item in valid:
        item_id = item.pop("id")
        result[str(item_id)] = item
    return result


def _mistral_restore_single_request(
    record: dict[str, Any],
    api_key: str,
    model: str,
    timeout: float,
) -> dict[str, Any] | None:
    system_prompt = (
        "You are an OCR correction assistant. Return only JSON with keys: "
        "restored_text, is_confident. Use the OCR variants to reconstruct the phrase. "
        "If not confident, set restored_text to empty string and is_confident=false."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "tesseract_text": (record.get("tesseract_text") or "")[:MAX_TEXT_CHARS],
                        "ocr_space_text": (record.get("ocr_space_text") or "")[:MAX_TEXT_CHARS],
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
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        response_json = json.loads(resp.read().decode("utf-8"))
    content = response_json["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed:
        return None
    return _normalize_restore_payload(parsed)


def _apply_mistral_restore(
    records: list[dict[str, Any]],
    api_key: str,
    model: str,
    invalid_log_path: Path | None,
    batch_size: int,
    timeout: float,
    request_sleep: float,
) -> dict[str, dict[str, Any]]:
    if not records:
        return {}
    results: dict[str, dict[str, Any]] = {}
    chunks = _chunked(records, batch_size)
    for chunk in chunks:
        try:
            batch = _mistral_restore_batch_request(
                chunk,
                api_key,
                model,
                invalid_log_path,
                timeout,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Mistral restore batch error: %s", exc)
            batch = None
        if not batch:
            LOGGER.warning("Mistral restore response invalid; falling back to per-item")
            for record in chunk:
                try:
                    item = _mistral_restore_single_request(record, api_key, model, timeout)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Mistral restore error for %s: %s", record.get("filename"), exc)
                    item = None
                if request_sleep > 0:
                    time.sleep(request_sleep)
                if not item:
                    continue
                results[str(record.get("filename"))] = item
            continue
        for record in chunk:
            item_id = str(record.get("filename"))
            item = batch.get(item_id)
            if not item:
                LOGGER.warning("Mistral restore missing result for %s", item_id)
                continue
            results[item_id] = item
        if request_sleep > 0:
            time.sleep(request_sleep)
    return results


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
        "You are a strict moderation assistant. Return only JSON with keys: "
        "is_sexual, is_profanity, is_politics, is_insults, is_threats, is_harassment, "
        "is_twitch_banned. Use true if a category is present or has clear signs. "
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
        "is_politics, is_insults, is_threats, is_harassment, is_twitch_banned. "
        "Use true if the category is present or has clear signs. "
        "Use false only if you are confident it is absent. "
        "If the text is empty or insufficient, return all censor flags true."
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
                text = record.get("text", "")
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
        with urlopen(req, timeout=timeout) as resp:
            response_json = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        LOGGER.warning("OCR.space HTTP %s: %s", exc.code, detail.strip())
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


def _upsert_by_filename(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    existing = _load_json_records(path) if path.exists() else []
    if existing is None:
        existing = []
    by_filename: dict[str, dict[str, Any]] = {}
    for item in existing:
        filename = item.get("filename")
        if isinstance(filename, str):
            by_filename[filename] = item
    for record in records:
        filename = record.get("filename")
        if isinstance(filename, str):
            by_filename[filename] = record
    merged = list(by_filename.values())
    _write_json(path, merged)


def _build_mistral_incorrect_path(path: Path, date_str: str) -> Path:
    return path.with_name(f"{path.stem}_mistral_incorrect_{date_str}{path.suffix}")


def _build_ocr_variants_path(path: Path, date_str: str) -> Path:
    return path.with_name(f"{path.stem}_ocr_variants_{date_str}{path.suffix}")


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


def _build_ocr_variants(
    records: list[dict[str, Any]],
    images_dir: Path,
    crop: tuple[float, float, float, float],
    ocrspace_cfg: dict[str, Any],
    ocrspace_sleep: float,
) -> list[dict[str, Any]]:
    image_index = _build_image_index(images_dir)
    results: list[dict[str, Any]] = []
    for record in records:
        filename = record.get("filename")
        if not isinstance(filename, str):
            continue
        image_path = image_index.get(filename)
        tesseract_text = record.get("text") or ""
        ocr_space_text = ""
        if not image_path:
            LOGGER.warning("Image not found for %s", filename)
        else:
            try:
                image_bytes = _image_to_jpeg_bytes(image_path, crop)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to prepare image %s: %s", filename, exc)
                image_bytes = b""
            if image_bytes:
                if ocrspace_cfg.get("api_key"):
                    try:
                        ocr_space_text = _ocr_space_text(
                            image_bytes,
                            ocrspace_cfg["api_key"],
                            ocrspace_cfg["api_url"],
                            ocrspace_cfg["language"],
                            ocrspace_cfg["engine"],
                            ocrspace_cfg["timeout"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("OCR.space failed for %s: %s", filename, exc)
                        ocr_space_text = ""
                    if ocrspace_sleep > 0:
                        time.sleep(ocrspace_sleep)
                else:
                    LOGGER.warning("OCR.space API key missing; skipping OCR.space for %s", filename)
        results.append(
            {
                "number": record.get("number"),
                "datetime": record.get("datetime"),
                "filename": filename,
                "tg_message_id": record.get("tg_message_id"),
                "tg_datetime_utc": record.get("tg_datetime_utc"),
                "tesseract_text": tesseract_text,
                "ocr_space_text": ocr_space_text,
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
        "--mistral-request-sleep",
        type=float,
        default=None,
        help="Sleep between Mistral requests (default: env MISTRAL_REQUEST_SLEEP or 0)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory with images for external OCR fallback (default: env OCR_IMAGES_DIR)",
    )
    parser.add_argument(
        "--ocrspace-api-key",
        default=None,
        help="OCR.space API key (default: env OCRSPACE_API_KEY)",
    )
    parser.add_argument(
        "--ocrspace-api-url",
        default=None,
        help="OCR.space API URL (default: env OCRSPACE_API_URL)",
    )
    parser.add_argument(
        "--ocrspace-language",
        default=None,
        help="OCR.space language (default: env OCRSPACE_LANGUAGE)",
    )
    parser.add_argument(
        "--ocrspace-engine",
        type=int,
        default=None,
        help="OCR.space OCR engine (default: env OCRSPACE_OCR_ENGINE or 2)",
    )
    parser.add_argument(
        "--ocrspace-timeout",
        type=float,
        default=None,
        help="OCR.space timeout (default: env OCRSPACE_TIMEOUT or 60)",
    )
    parser.add_argument(
        "--ocrspace-request-sleep",
        type=float,
        default=None,
        help="Sleep between OCR.space requests (default: env OCRSPACE_REQUEST_SLEEP or 0)",
    )
    parser.add_argument(
        "--external-ocr-crop",
        default=None,
        help="External OCR crop as left,top,right,bottom (default: env EXTERNAL_OCR_CROP)",
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
    mistral_request_sleep = _resolve_nonnegative_float(
        args.mistral_request_sleep,
        "MISTRAL_REQUEST_SLEEP",
        DEFAULT_MISTRAL_SLEEP,
    )
    invalid_log_path = Path(
        args.invalid_log_file or os.getenv("MISTRAL_INVALID_LOG_FILE", "data/mistral_invalid.log")
    )
    images_dir_value = args.images_dir or os.getenv("OCR_IMAGES_DIR")
    ocrspace_api_key = args.ocrspace_api_key or os.getenv("OCRSPACE_API_KEY")
    ocrspace_api_url = args.ocrspace_api_url or os.getenv(
        "OCRSPACE_API_URL", DEFAULT_OCRSPACE_URL
    )
    ocrspace_language = args.ocrspace_language or os.getenv(
        "OCRSPACE_LANGUAGE",
        DEFAULT_OCRSPACE_LANGUAGE,
    )
    raw_engine = args.ocrspace_engine or os.getenv("OCRSPACE_OCR_ENGINE")
    try:
        ocrspace_engine = int(raw_engine) if raw_engine else DEFAULT_OCRSPACE_ENGINE
    except ValueError:
        ocrspace_engine = DEFAULT_OCRSPACE_ENGINE
    if ocrspace_engine not in {1, 2}:
        ocrspace_engine = DEFAULT_OCRSPACE_ENGINE
    ocrspace_timeout = _resolve_float(
        args.ocrspace_timeout,
        "OCRSPACE_TIMEOUT",
        DEFAULT_OCRSPACE_TIMEOUT,
    )
    ocrspace_request_sleep = _resolve_nonnegative_float(
        args.ocrspace_request_sleep,
        "OCRSPACE_REQUEST_SLEEP",
        DEFAULT_OCRSPACE_SLEEP,
    )
    crop_raw = args.external_ocr_crop or os.getenv("EXTERNAL_OCR_CROP", DEFAULT_EXTERNAL_CROP)
    try:
        external_crop = _parse_crop(crop_raw)
    except ValueError as exc:
        LOGGER.warning("Invalid EXTERNAL_OCR_CROP: %s", exc)
        return 2

    files = [Path(p) for p in args.json_files]
    if args.input_dir:
        input_dir = Path(args.input_dir)
        files.extend(sorted(input_dir.glob(args.pattern)))
    files = [path for path in files if path.exists()]
    if not files:
        LOGGER.warning("No JSON files to process.")
        return 1

    for path in files:
        if (
            "_ocr_failed_" in path.stem
            or "_ocr_variants_" in path.stem
            or "_mistral_incorrect_" in path.stem
        ):
            LOGGER.info("Skipping auxiliary file %s", path)
            continue
        records = _load_json_records(path)
        if records is None:
            LOGGER.warning("Invalid JSON format in %s", path)
            continue
        LOGGER.info("Moderating %s (%s records)", path, len(records))
        _apply_mistral(
            records,
            api_key,
            model,
            invalid_log_path,
            batch_size,
            mistral_timeout,
            mistral_request_sleep,
        )
        _write_json(path, records)
        LOGGER.info("Updated %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
