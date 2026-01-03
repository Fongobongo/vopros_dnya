import argparse
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
import pytesseract


LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30
VOPROS_RE = re.compile(
    r"(?:vopros(?:[._\\s]*dna)?|_dna|\\.dna|\\bdna\\b)",
    re.IGNORECASE,
)
THUMB_RE = re.compile(r"^(?P<base>.+)_thumb(?P<ext>\.(?:jpg|jpeg|png|webp))$", re.IGNORECASE)
DEFAULT_DETECT_FRACTION = 0.25
DEFAULT_CROP_FRACTION = 0.25
DEFAULT_EXTRA_TOP_PX = 23


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


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_thumb(path: Path) -> bool:
    return bool(THUMB_RE.match(path.name))


def _normalize_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_vopros(path: Path, detect_fraction: float, lang: str, psm: int) -> bool:
    image = Image.open(path).convert("L")
    width, height = image.size
    crop_top = int(height * (1.0 - detect_fraction))
    crop = image.crop((0, crop_top, width, height))
    if max(crop.size) > 900:
        crop.thumbnail((900, 900))
    text = pytesseract.image_to_string(
        crop,
        lang=lang,
        config=f"--psm {psm} --oem 1",
    )
    cleaned = _normalize_text(text).lower()
    return bool(VOPROS_RE.search(cleaned))


def _crop_vertical(
    src: Path,
    dest: Path,
    top_fraction: float,
    bottom_fraction: float,
    extra_top_px: int,
) -> None:
    image = Image.open(src).convert("RGB")
    width, height = image.size
    top_px = int(height * top_fraction) + max(0, extra_top_px)
    bottom_px = int(height * bottom_fraction)
    if top_px + bottom_px >= height:
        raise ValueError("Crop fractions too large for image height.")
    cropped = image.crop((0, top_px, width, height - bottom_px))
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        cropped.save(dest, format="JPEG", quality=92)
    elif suffix == ".png":
        cropped.save(dest, format="PNG")
    elif suffix == ".webp":
        cropped.save(dest, format="WEBP", quality=92)
    else:
        cropped.save(dest)


def _remove_thumbs(target_dir: Path) -> int:
    if not target_dir.exists():
        return 0
    removed = 0
    for path in target_dir.rglob("*"):
        if not path.is_file():
            continue
        if _is_thumb(path):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                LOGGER.warning("Failed to remove thumb %s: %s", path.name, exc)
    return removed


def _iter_images(input_dir: Path, files: list[Path] | None) -> list[Path]:
    if files:
        result = []
        for item in files:
            if item.is_absolute():
                path = item
            else:
                path = input_dir / item
                if not path.exists():
                    fallback = input_dir / item.name
                    if fallback.exists():
                        path = fallback
            if path.exists() and path.is_file():
                result.append(path)
        return result
    result = []
    for path in sorted(input_dir.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if _is_thumb(path):
            continue
        result.append(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect question images and crop them for OCR.",
    )
    parser.add_argument("input_dir", help="Directory with images")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional list of image files to process (relative to input_dir)",
    )
    parser.add_argument(
        "--not-a-question-dir",
        default=None,
        help="Directory to copy non-question images (default: data/not_question)",
    )
    parser.add_argument(
        "--cropped-dir",
        default=None,
        help="Directory to save cropped images (default: data/cropped)",
    )
    parser.add_argument(
        "--detect-fraction",
        type=float,
        default=DEFAULT_DETECT_FRACTION,
        help="Bottom fraction to scan for vopros text (default: 0.25)",
    )
    parser.add_argument(
        "--crop-top",
        type=float,
        default=DEFAULT_CROP_FRACTION,
        help="Top fraction to remove when cropping (default: 0.25)",
    )
    parser.add_argument(
        "--crop-bottom",
        type=float,
        default=DEFAULT_CROP_FRACTION,
        help="Bottom fraction to remove when cropping (default: 0.25)",
    )
    parser.add_argument(
        "--crop-top-px",
        type=int,
        default=DEFAULT_EXTRA_TOP_PX,
        help="Extra pixels to remove from top (default: 23)",
    )
    parser.add_argument(
        "--lang",
        default="eng+rus",
        help="Tesseract languages (default: eng+rus)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6)",
    )
    parser.add_argument(
        "--manifest-file",
        default=None,
        help="Manifest JSON path (default: env PREPROCESS_MANIFEST_FILE)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if manifest is up to date",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/preprocess.log)",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    log_file = args.log_file or os.getenv("PREPROCESS_LOG_FILE") or "data/preprocess.log"
    _setup_logging(Path(log_file))

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        LOGGER.warning("Input directory not found: %s", input_dir)
        return 1
    not_a_question_dir = Path(
        args.not_a_question_dir
        or os.getenv("PREPROCESS_NOT_A_QUESTION_DIR")
        or "data/not_question"
    )
    cropped_dir = Path(
        args.cropped_dir
        or os.getenv("PREPROCESS_CROPPED_DIR")
        or "data/cropped"
    )
    manifest_path = Path(
        args.manifest_file
        or os.getenv("PREPROCESS_MANIFEST_FILE")
        or "data/preprocess_manifest.json"
    )
    manifest = _load_manifest(manifest_path)

    removed_thumbs = _remove_thumbs(input_dir)
    removed_thumbs += _remove_thumbs(not_a_question_dir)
    removed_thumbs += _remove_thumbs(cropped_dir)
    if removed_thumbs:
        LOGGER.info("Removed %s thumbnail files", removed_thumbs)

    files = _iter_images(input_dir, [Path(p) for p in args.files] if args.files else None)
    if not files:
        LOGGER.info("No images to process in %s", input_dir)
        return 0

    processed = 0
    cropped = 0
    copied = 0
    skipped = 0

    for path in files:
        if not path.exists():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        entry = manifest.get(path.name, {})
        if (
            not args.force
            and entry.get("mtime") == mtime
            and entry.get("status") == "question"
            and (cropped_dir / path.name).exists()
        ):
            skipped += 1
            continue
        if (
            not args.force
            and entry.get("mtime") == mtime
            and entry.get("status") == "not_a_question"
            and (not_a_question_dir / path.name).exists()
        ):
            skipped += 1
            continue
        try:
            has_vopros = _detect_vopros(path, args.detect_fraction, args.lang, args.psm)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Detection failed for %s: %s", path.name, exc)
            skipped += 1
            continue

        if not has_vopros:
            dest = not_a_question_dir / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, dest)
            except OSError as exc:
                LOGGER.warning("Copy failed for %s: %s", path.name, exc)
                skipped += 1
                continue
            cropped_path = cropped_dir / path.name
            if cropped_path.exists():
                try:
                    cropped_path.unlink()
                except OSError as exc:
                    LOGGER.warning("Failed to remove stale crop %s: %s", cropped_path.name, exc)
            manifest[path.name] = {
                "status": "not_a_question",
                "mtime": mtime,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            copied += 1
            processed += 1
            LOGGER.info("Copied %s to %s", dest.name, not_a_question_dir)
            continue

        try:
            cropped_path = cropped_dir / path.name
            _crop_vertical(path, cropped_path, args.crop_top, args.crop_bottom, args.crop_top_px)
            not_question_path = not_a_question_dir / path.name
            if not_question_path.exists():
                try:
                    not_question_path.unlink()
                except OSError as exc:
                    LOGGER.warning(
                        "Failed to remove stale not-question copy %s: %s",
                        not_question_path.name,
                        exc,
                    )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Crop failed for %s: %s", path.name, exc)
            skipped += 1
            continue

        manifest[path.name] = {
            "status": "question",
            "mtime": path.stat().st_mtime,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        cropped += 1
        processed += 1
        LOGGER.info("Cropped %s", path.name)

    _save_manifest(manifest_path, manifest)
    LOGGER.info(
        "Preprocess done processed=%s cropped=%s copied=%s skipped=%s",
        processed,
        cropped,
        copied,
        skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
