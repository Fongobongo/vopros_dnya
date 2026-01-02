import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image
import pytesseract


FILENAME_RE = re.compile(r"photo_(\d+)@(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})")


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


def _crop_center(image: Image.Image, crop: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left_pct, top_pct, right_pct, bottom_pct = crop
    left = int(width * left_pct)
    top = int(height * top_pct)
    right = int(width * (1.0 - right_pct))
    bottom = int(height * (1.0 - bottom_pct))
    return image.crop((left, top, right, bottom))


def _clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bvopros[_\s]*dna\b", "", text, flags=re.IGNORECASE)
    return text.strip(" -–—")


def _extract_text(
    path: Path,
    crop: tuple[float, float, float, float],
    lang: str,
    psm: int,
) -> str:
    image = Image.open(path).convert("L")
    image = _crop_center(image, crop)
    if max(image.size) > 900:
        image.thumbnail((900, 900))
    text = pytesseract.image_to_string(
        image,
        lang=lang,
        config=f"--psm {psm} --oem 1",
    )
    return _clean_text(text)


def extract_text(
    path: Path,
    crop: tuple[float, float, float, float],
    lang: str,
    psm: int,
) -> str:
    return _extract_text(path, crop, lang, psm)


def _process_one(
    path_str: str,
    crop: tuple[float, float, float, float],
    lang: str,
    psm: int,
) -> tuple[str, str]:
    path = Path(path_str)
    text = _extract_text(path, crop, lang, psm)
    return path_str, text


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract questions from center of images.")
    parser.add_argument("input_dir", help="Directory with images")
    parser.add_argument(
        "--out",
        default="data/questions.json",
        help="Output JSON file",
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
        "--batch",
        type=int,
        default=50,
        help="Write output every N items (default: 50)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    crop_parts = [float(p) for p in args.crop.split(",")]
    if len(crop_parts) != 4:
        print("Invalid crop format. Use: left,top,right,bottom")
        return 1
    crop = tuple(crop_parts)  # type: ignore[assignment]

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

    images = []
    for name, full_path in full_images.items():
        chosen = thumb_images.get(name, full_path)
        images.append((name, chosen))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    seen_filenames = set()
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    if isinstance(item, dict) and item.get("filename"):
                        seen_filenames.add(item["filename"])
            else:
                existing = []
        except json.JSONDecodeError:
            existing = []
    results = list(existing)

    meta_by_path = {}
    todo = []
    for filename, path in images:
        if filename in seen_filenames:
            continue
        number, dt_str = _parse_metadata(filename)
        meta_by_path[str(path)] = {
            "number": number,
            "datetime": dt_str,
            "filename": filename,
        }
        todo.append(path)
    total = len(todo)
    if total == 0:
        print("No новых файлов для обработки.")
        return 0

    workers = max(1, args.workers)
    batch = max(1, args.batch)
    processed = 0
    fallback_index = len(results) + 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_one, str(path), crop, args.lang, args.psm): path
            for path in todo
        }
        for future in as_completed(futures):
            path_str, text = future.result()
            meta = meta_by_path.get(path_str, {})
            number = meta.get("number")
            if not number:
                number = fallback_index
                fallback_index += 1
            results.append(
                {
                    "number": number,
                    "datetime": meta.get("datetime"),
                    "filename": meta.get("filename") or Path(path_str).name,
                    "text": text,
                    "llm_validated": False,
                    "human_validated": False,
                }
            )
            processed += 1
            if processed % batch == 0 or processed == total:
                out_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"Processed {processed}/{total}...")

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} items to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
