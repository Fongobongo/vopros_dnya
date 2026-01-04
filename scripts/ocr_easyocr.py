import argparse
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import easyocr


LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
DEFAULT_LANGS = "ru,en"
DEFAULT_WORKERS = 2
FILENAME_RE = re.compile(r"photo_(\d+)@(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})")
THUMB_RE = re.compile(r".+_thumb\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
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

_READER_LOCAL = threading.local()


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


def _resolve_bool(explicit_true: bool, explicit_false: bool, env_key: str, default: bool) -> bool:
    if explicit_true:
        return True
    if explicit_false:
        return False
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


def _get_reader(langs: list[str], gpu: bool) -> easyocr.Reader:
    reader = getattr(_READER_LOCAL, "reader", None)
    if reader is None:
        reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
        _READER_LOCAL.reader = reader
    return reader


def _read_text(path: Path, langs: list[str], gpu: bool) -> str:
    reader = _get_reader(langs, gpu)
    pieces = reader.readtext(str(path), detail=0, paragraph=True)
    if isinstance(pieces, list):
        text = " ".join(str(item) for item in pieces)
    else:
        text = str(pieces or "")
    return _clean_text(text)


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


def _default_flags() -> dict[str, Any]:
    return {flag: None for flag in FLAGS}


def _list_images(input_dir: Path) -> list[dict[str, Any]]:
    items = []
    for path in sorted(input_dir.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if THUMB_RE.match(path.name):
            continue
        items.append({"filename": path.name, "path": path})
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


def _ocr_items(
    items: list[dict[str, Any]],
    langs: list[str],
    gpu: bool,
    workers: int,
) -> list[dict[str, Any]]:
    results = []
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_read_text, item["path"], langs, gpu): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                text = future.result()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("OCR failed for %s: %s", item["filename"], exc)
                text = ""
            if isinstance(text, str):
                text = text.strip() or None
            else:
                text = None
            number, dt_str = _parse_metadata(item["filename"])
            record = {
                "number": number,
                "datetime": dt_str,
                "filename": item["filename"],
                "text": text,
                "llm_validated": None,
                "human_validated": None,
                "is_correct": None,
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
        description="OCR images from a directory using EasyOCR and write per-date JSON files.",
    )
    parser.add_argument("input_dir", help="Directory with images")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON base path, supports {date} (default: data/questions_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--langs",
        default=None,
        help="EasyOCR languages, e.g. ru,en (default: env EASYOCR_LANGS or ru,en)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for EasyOCR (default: env EASYOCR_GPU or false)",
    )
    group.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for EasyOCR (overrides --gpu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel OCR workers (default: env EASYOCR_WORKERS or 2)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/easyocr.log)",
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
    log_file = args.log_file or os.getenv("EASYOCR_LOG_FILE") or "data/easyocr.log"
    _setup_logging(Path(log_file))

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        LOGGER.warning("Input directory not found: %s", input_dir)
        return 1

    lang_spec = args.langs or os.getenv("EASYOCR_LANGS", DEFAULT_LANGS)
    langs = _parse_langs(lang_spec)
    if not langs:
        LOGGER.warning("No EasyOCR languages configured.")
        return 1
    gpu = _resolve_bool(args.gpu, args.cpu, "EASYOCR_GPU", False)
    workers = _resolve_int(args.workers, "EASYOCR_WORKERS", DEFAULT_WORKERS)

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

    records = _ocr_items(todo, langs, gpu, workers)
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
