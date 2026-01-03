import argparse
import io
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import easyocr
from PIL import Image
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.extract_questions import extract_text


LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
USER_AGENT = "Mozilla/5.0 (compatible; OCRFetcher/1.0; +https://t.me)"
MAX_TEXT_CHARS = 4000
DEFAULT_MISTRAL_TIMEOUT = 60.0
DEFAULT_MISTRAL_SLEEP = 0.0
DEFAULT_OCRSPACE_API_URL = "https://api.ocr.space/parse/image"
DEFAULT_OCRSPACE_ENGINE = 2
DEFAULT_OCRSPACE_LANGUAGE = "rus"
DEFAULT_OCRSPACE_TIMEOUT = 60.0
DEFAULT_OCRSPACE_SLEEP = 0.0
DEFAULT_EASYOCR_LANGS = "ru,en"
DEFAULT_EASYOCR_WORKERS = 2
FILENAME_RE = re.compile(r"photo_(\d+)@(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})")
FLAGS = [
    "is_sexual",
    "is_profanity",
    "is_politics",
    "is_insults",
    "is_threats",
    "is_harassment",
    "is_twitch_banned",
]

_EASYOCR_LOCAL = threading.local()


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


def _parse_langs(value: str) -> list[str]:
    parts = re.split(r"[,+]", value)
    return [part.strip() for part in parts if part.strip()]


def _clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bvopros[_\s]*dna\b", "", text, flags=re.IGNORECASE)
    return text.strip(" -–—")


def _extract_json(content: str) -> dict[str, Any] | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None

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


def _get_easyocr_reader(langs: list[str], gpu: bool) -> easyocr.Reader:
    reader = getattr(_EASYOCR_LOCAL, "reader", None)
    if reader is None:
        reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
        _EASYOCR_LOCAL.reader = reader
    return reader


def _easyocr_text(path: Path, langs: list[str], gpu: bool) -> str:
    reader = _get_easyocr_reader(langs, gpu)
    pieces = reader.readtext(str(path), detail=0, paragraph=True)
    if isinstance(pieces, list):
        text = " ".join(str(item) for item in pieces)
    else:
        text = str(pieces or "")
    return _clean_text(text)


def _image_to_jpeg_bytes(path: Path) -> bytes:
    image = Image.open(path).convert("RGB")
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
    path: Path,
    api_key: str,
    api_url: str,
    language: str,
    engine: int,
    timeout: float,
) -> str:
    image_bytes = _image_to_jpeg_bytes(path)
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
        return _clean_text(parsed_text)
    return ""


def _mistral_restore_text(
    tesseract_text: str,
    easyocr_text: str,
    ocrspace_text: str,
    api_key: str,
    model: str,
    timeout: float,
) -> str:
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
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        response_json = json.loads(resp.read().decode("utf-8"))
    content = response_json["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed:
        return ""
    restored_text = parsed.get("restored_text")
    if not isinstance(restored_text, str):
        return ""
    return restored_text.strip()


def _consensus_text(values: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
        if counts[value] >= 2:
            return value
    return None


def _resolve_easyocr_settings(
    langs_override: str | None,
    gpu_override: bool | None,
    cpu_override: bool | None,
    workers_override: int | None,
) -> tuple[list[str], bool, int]:
    langs_raw = langs_override or os.getenv("EASYOCR_LANGS", DEFAULT_EASYOCR_LANGS)
    langs = _parse_langs(langs_raw)
    if not langs:
        raise ValueError("EasyOCR languages not configured")
    if gpu_override is True:
        gpu = True
    elif cpu_override is True:
        gpu = False
    else:
        gpu = _resolve_bool(None, "EASYOCR_GPU", False)
    workers = _resolve_int(workers_override, "EASYOCR_WORKERS", DEFAULT_EASYOCR_WORKERS)
    return langs, gpu, workers


def _resolve_ocrspace_config() -> tuple[dict[str, Any], float]:
    cfg = {
        "api_key": os.getenv("OCRSPACE_API_KEY"),
        "api_url": os.getenv("OCRSPACE_API_URL", DEFAULT_OCRSPACE_API_URL),
        "language": os.getenv("OCRSPACE_LANGUAGE", DEFAULT_OCRSPACE_LANGUAGE),
        "engine": os.getenv("OCRSPACE_OCR_ENGINE", str(DEFAULT_OCRSPACE_ENGINE)),
        "timeout": os.getenv("OCRSPACE_TIMEOUT", str(DEFAULT_OCRSPACE_TIMEOUT)),
    }
    request_sleep = _resolve_nonnegative_float(
        None,
        "OCRSPACE_REQUEST_SLEEP",
        DEFAULT_OCRSPACE_SLEEP,
    )
    return cfg, request_sleep


def _resolve_mistral_config() -> tuple[dict[str, Any], float]:
    cfg = {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        "timeout": os.getenv("MISTRAL_TIMEOUT", str(DEFAULT_MISTRAL_TIMEOUT)),
    }
    request_sleep = _resolve_nonnegative_float(
        None,
        "MISTRAL_REQUEST_SLEEP",
        DEFAULT_MISTRAL_SLEEP,
    )
    return cfg, request_sleep

def _parse_crop(value: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Invalid crop format. Use: left,top,right,bottom")
    return tuple(parts)  # type: ignore[return-value]


def _parse_metadata(name: str) -> tuple[int | None, str | None]:
    match = FILENAME_RE.search(name)
    if not match:
        return None, None
    number = int(match.group(1))
    date_part = match.group(2)
    time_part = match.group(3)
    try:
        dt = datetime.strptime(f"{date_part} {time_part}", "%d-%m-%Y %H-%M-%S")
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        dt_str = None
    return number, dt_str


def _date_from_value(value: str | None, fallback: datetime) -> str:
    if isinstance(value, str):
        if " " in value:
            return value.split(" ")[0]
        if "T" in value:
            return value.split("T")[0]
    return fallback.strftime("%Y-%m-%d")


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


def _default_flags() -> dict[str, bool]:
    return {flag: True for flag in FLAGS}


def _list_images(input_dir: Path) -> list[dict[str, Any]]:
    full_images: dict[str, Path] = {}
    thumb_images: dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
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


def _run_dual_ocr(
    items: list[dict[str, Any]],
    crop: tuple[float, float, float, float],
    lang: str,
    psm: int,
    tesseract_workers: int,
    easyocr_langs: list[str],
    easyocr_gpu: bool,
    easyocr_workers: int,
) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {
        item["filename"]: {"tesseract_text": "", "easyocr_text": ""} for item in items
    }
    total_workers = max(1, tesseract_workers + easyocr_workers)
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {}
        for item in items:
            futures[
                executor.submit(extract_text, item["path"], crop, lang, psm)
            ] = (item["filename"], "tesseract")
            futures[
                executor.submit(_easyocr_text, item["path"], easyocr_langs, easyocr_gpu)
            ] = (item["filename"], "easyocr")
        for future in as_completed(futures):
            filename, kind = futures[future]
            try:
                text = future.result()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("%s OCR failed for %s: %s", kind, filename, exc)
                text = ""
            results[filename][f"{kind}_text"] = text
    return results


def _ocr_items(
    items: list[dict[str, Any]],
    crop: tuple[float, float, float, float],
    lang: str,
    psm: int,
    workers: int,
    easyocr_langs: list[str],
    easyocr_gpu: bool,
    easyocr_workers: int,
    ocrspace_cfg: dict[str, Any],
    ocrspace_request_sleep: float,
    mistral_cfg: dict[str, Any],
    mistral_request_sleep: float,
) -> list[dict[str, Any]]:
    results = []
    if not items:
        return results
    dual_results = _run_dual_ocr(
        items,
        crop,
        lang,
        psm,
        workers,
        easyocr_langs,
        easyocr_gpu,
        easyocr_workers,
    )
    ocrspace_api_key = ocrspace_cfg.get("api_key")
    ocrspace_api_url = ocrspace_cfg.get("api_url")
    ocrspace_language = ocrspace_cfg.get("language")
    ocrspace_engine = ocrspace_cfg.get("engine")
    ocrspace_timeout = ocrspace_cfg.get("timeout")

    mistral_api_key = mistral_cfg.get("api_key")
    mistral_model = mistral_cfg.get("model")
    mistral_timeout = mistral_cfg.get("timeout")

    for item in items:
        filename = item["filename"]
        variants = dual_results.get(filename, {})
        tesseract_text = variants.get("tesseract_text", "")
        easyocr_text = variants.get("easyocr_text", "")
        ocrspace_text = ""
        mistral_text = ""
        text = ""
        is_correct = False

        if tesseract_text and tesseract_text == easyocr_text:
            text = tesseract_text
            is_correct = True
        else:
            need_ocrspace = tesseract_text != easyocr_text or not tesseract_text
            if need_ocrspace and ocrspace_api_key:
                try:
                    ocrspace_text = _ocr_space_text(
                        Path(item["path"]),
                        ocrspace_api_key,
                        ocrspace_api_url or DEFAULT_OCRSPACE_API_URL,
                        ocrspace_language or DEFAULT_OCRSPACE_LANGUAGE,
                        int(ocrspace_engine or DEFAULT_OCRSPACE_ENGINE),
                        float(ocrspace_timeout or DEFAULT_OCRSPACE_TIMEOUT),
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("OCR.space failed for %s: %s", filename, exc)
                if ocrspace_request_sleep > 0:
                    time.sleep(ocrspace_request_sleep)
            elif need_ocrspace and not ocrspace_api_key:
                LOGGER.warning("OCR.space API key not set; skipping for %s", filename)

            consensus = _consensus_text([tesseract_text, easyocr_text, ocrspace_text])
            if consensus:
                text = consensus
                is_correct = True
            else:
                if mistral_api_key:
                    try:
                        mistral_text = _mistral_restore_text(
                            tesseract_text,
                            easyocr_text,
                            ocrspace_text,
                            mistral_api_key,
                            mistral_model or "mistral-small-latest",
                            float(mistral_timeout or DEFAULT_MISTRAL_TIMEOUT),
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Mistral restore failed for %s: %s", filename, exc)
                    if mistral_request_sleep > 0:
                        time.sleep(mistral_request_sleep)
                else:
                    LOGGER.warning("MISTRAL_API_KEY not set; skipping restore for %s", filename)
                consensus = _consensus_text(
                    [tesseract_text, easyocr_text, ocrspace_text, mistral_text]
                )
                if consensus:
                    text = consensus
                    is_correct = True

        number, dt_str = _parse_metadata(filename)
        record = {
            "number": number,
            "datetime": dt_str,
            "filename": filename,
            "text": text if is_correct else "",
            "tesseract_text": tesseract_text,
            "easyocr_text": easyocr_text,
            "ocrspace_text": ocrspace_text,
            "mistral_text": mistral_text,
            "llm_validated": False,
            "human_validated": False,
            "is_correct": is_correct,
            "tg_message_id": item.get("tg_message_id"),
            "tg_datetime_utc": item.get("tg_datetime_utc"),
        }
        record.update(_default_flags())
        results.append(record)
    results.sort(key=lambda item: item["filename"])
    fallback_index = 1
    for record in results:
        if not record.get("number"):
            record["number"] = fallback_index
            fallback_index += 1
    return results


def build_records(
    items: list[dict[str, Any]],
    crop: tuple[float, float, float, float],
    lang: str,
    psm: int,
    workers: int,
    easyocr_langs: str | None = None,
    easyocr_gpu: bool | None = None,
    easyocr_cpu: bool | None = None,
    easyocr_workers: int | None = None,
) -> list[dict[str, Any]]:
    try:
        langs, gpu, easy_workers = _resolve_easyocr_settings(
            easyocr_langs,
            easyocr_gpu,
            easyocr_cpu,
            easyocr_workers,
        )
    except ValueError:
        LOGGER.warning("EasyOCR languages not configured.")
        return []
    ocrspace_cfg, ocrspace_request_sleep = _resolve_ocrspace_config()
    mistral_cfg, mistral_request_sleep = _resolve_mistral_config()
    return _ocr_items(
        items,
        crop,
        lang,
        psm,
        workers,
        langs,
        gpu,
        easy_workers,
        ocrspace_cfg,
        ocrspace_request_sleep,
        mistral_cfg,
        mistral_request_sleep,
    )


def _ocr_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    extracted = 0
    for record in records:
        text = record.get("text") or ""
        if isinstance(text, str) and text.strip():
            extracted += 1
    total = len(records)
    failed = total - extracted
    return {"total": total, "extracted": extracted, "failed": failed}


def _group_by_date(records: list[dict[str, Any]], now: datetime) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dt_str = record.get("tg_datetime_utc") or record.get("datetime")
        date_str = _date_from_value(dt_str, now)
        grouped.setdefault(date_str, []).append(record)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OCR images from a directory and write per-date JSON files.",
    )
    parser.add_argument("input_dir", help="Directory with images")
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
        "--easyocr-langs",
        default=None,
        help="EasyOCR languages (default: env EASYOCR_LANGS or ru,en)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--easyocr-gpu",
        action="store_true",
        help="Use GPU for EasyOCR (default: env EASYOCR_GPU or false)",
    )
    group.add_argument(
        "--easyocr-cpu",
        action="store_true",
        help="Force CPU for EasyOCR",
    )
    parser.add_argument(
        "--easyocr-workers",
        type=int,
        default=None,
        help="EasyOCR workers (default: env EASYOCR_WORKERS or 2)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/ocr.log)",
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
    log_file = args.log_file or os.getenv("OCR_LOG_FILE") or "data/ocr.log"
    _setup_logging(Path(log_file))

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

    try:
        easyocr_langs, easyocr_gpu, easyocr_workers = _resolve_easyocr_settings(
            args.easyocr_langs,
            args.easyocr_gpu,
            args.easyocr_cpu,
            args.easyocr_workers,
        )
    except ValueError:
        LOGGER.warning("EasyOCR languages not configured.")
        return 1
    ocrspace_cfg, ocrspace_request_sleep = _resolve_ocrspace_config()
    mistral_cfg, mistral_request_sleep = _resolve_mistral_config()

    manifest = _load_manifest(Path(manifest_file))
    for item in items:
        meta = manifest.get(item["filename"], {})
        item["tg_message_id"] = meta.get("message_id")
        item["tg_datetime_utc"] = meta.get("message_datetime_utc")
        _, dt_str = _parse_metadata(item["filename"])
        item["date"] = _date_from_value(item["tg_datetime_utc"] or dt_str, now)

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

    crop = _parse_crop(args.crop)
    records = _ocr_items(
        todo,
        crop,
        args.lang,
        args.psm,
        args.workers,
        easyocr_langs,
        easyocr_gpu,
        easyocr_workers,
        ocrspace_cfg,
        ocrspace_request_sleep,
        mistral_cfg,
        mistral_request_sleep,
    )
    if not records:
        return 0

    grouped = _group_by_date(records, now)
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
