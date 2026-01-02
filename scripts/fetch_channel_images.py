import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; OCRFetcher/1.0; +https://t.me)"
PHOTO_CLASSES = {
    "tgme_widget_message_photo_wrap",
    "tgme_widget_message_photo",
    "tgme_widget_message_video_thumb",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LOGGER = logging.getLogger(__name__)
LOG_RETENTION_DAYS = 30


def _parse_message_id(data_post: str) -> int | None:
    parts = data_post.split("/")
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _format_utc(dt: datetime | None) -> str | None:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _extract_background_url(style: str) -> str | None:
    match = re.search(r"background-image:url\\(['\\\"]?(.*?)['\\\"]?\\)", style)
    if match:
        return match.group(1)
    return None


class TgPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v for k, v in attrs}
        classes = set((attrs_dict.get("class") or "").split())

        if tag == "div" and "tgme_widget_message" in classes and "data-post" in attrs_dict:
            msg_id = _parse_message_id(attrs_dict["data-post"] or "")
            if msg_id is None:
                return
            self._current = {"id": msg_id, "datetime": None, "images": []}
            self.messages.append(self._current)
            return

        if not self._current:
            return

        if tag == "time" and "datetime" in attrs_dict:
            self._current["datetime"] = attrs_dict.get("datetime")
            return

        if classes & PHOTO_CLASSES:
            url = _extract_background_url(attrs_dict.get("style") or "")
            if url:
                self._current["images"].append(url)
            return

        if tag == "img" and "tgme_widget_message_photo" in classes:
            src = attrs_dict.get("src")
            if src:
                self._current["images"].append(src)


def _fetch_page(channel: str, before_id: int | None) -> str:
    url = f"https://t.me/s/{channel}"
    if before_id:
        url = f"{url}?before={before_id}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8")


def _download_image(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp, dest.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)


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


def _date_from_value(value: str | None, fallback: datetime) -> str:
    if isinstance(value, str):
        if " " in value:
            return value.split(" ")[0]
        if "T" in value:
            return value.split("T")[0]
    return fallback.strftime("%Y-%m-%d")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


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


def fetch_channel_images(
    channel: str,
    out_dir: Path,
    state_file: Path,
    since_hours: int = 24,
    max_pages: int = 10,
    backfill: bool = False,
    page_sleep: float = 1.0,
    download_sleep: float = 0.3,
) -> dict[str, list[dict[str, Any]]]:
    page_sleep = max(0.0, page_sleep)
    download_sleep = max(0.0, download_sleep)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_file)

    last_seen_id = state.get("last_seen_id")
    backfill_before_id = state.get("backfill_before_id") if backfill else None
    backfill_complete_state = bool(state.get("backfill_complete", False))
    now = datetime.now(timezone.utc)
    since_dt = None
    if not backfill and not last_seen_id:
        since_dt = now - timedelta(hours=max(1, since_hours))
    last_seen_filter = None if backfill else last_seen_id

    downloaded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    max_id_seen = last_seen_id or 0
    before_id = backfill_before_id if backfill_before_id else None
    oldest_id_seen: int | None = None
    backfill_complete = False
    LOGGER.info(
        "Fetch start channel=%s backfill=%s max_pages=%s before_id=%s",
        channel,
        backfill,
        max_pages,
        before_id,
    )

    for _ in range(max(1, max_pages)):
        html = _fetch_page(channel, before_id)
        parser = TgPageParser()
        parser.feed(html)
        messages = parser.messages
        if not messages:
            if backfill and before_id is not None:
                backfill_complete = True
            break

        min_id = min(msg["id"] for msg in messages)
        if oldest_id_seen is None or min_id < oldest_id_seen:
            oldest_id_seen = min_id
        max_id_seen = max(max_id_seen, max(msg["id"] for msg in messages))

        for msg in messages:
            msg_id = msg["id"]
            if last_seen_filter and msg_id <= last_seen_filter:
                continue
            msg_dt_utc = _format_utc(_parse_datetime(msg.get("datetime")))
            if since_dt:
                msg_dt = _parse_datetime(msg.get("datetime"))
                if msg_dt and msg_dt < since_dt:
                    continue
            for idx, url in enumerate(msg.get("images", []), start=1):
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                ext = Path(urlparse(url).path).suffix.lower()
                if ext not in IMAGE_EXTS:
                    continue
                filename = f"tg_{msg_id}_{idx}{ext}"
                dest = out_dir / filename
                if dest.exists():
                    continue
                try:
                    _download_image(url, dest)
                    downloaded.append(
                        {
                            "path": dest,
                            "message_id": msg_id,
                            "message_datetime_utc": msg_dt_utc,
                        }
                    )
                    LOGGER.info("Downloaded %s", dest.name)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to download %s: %s", url, exc)
                    failed.append(
                        {
                            "url": url,
                            "message_id": msg_id,
                            "message_datetime_utc": msg_dt_utc,
                        }
                    )
                if download_sleep > 0:
                    time.sleep(download_sleep)

        stop = False
        if last_seen_filter and min_id <= last_seen_filter:
            stop = True
        if since_dt:
            dts = [_parse_datetime(msg.get("datetime")) for msg in messages]
            dts = [dt for dt in dts if dt]
            if dts and min(dts) < since_dt:
                stop = True
        if backfill and before_id and min_id >= before_id:
            stop = True
            backfill_complete = True
        if stop:
            break
        before_id = min_id
        if page_sleep > 0:
            time.sleep(page_sleep)

    state_update = dict(state)
    state_update.update(
        {
            "last_seen_id": max_id_seen,
            "last_run_utc": now.isoformat(),
            "channel": channel,
        }
    )
    if backfill:
        if backfill_complete:
            state_update["backfill_before_id"] = None
            state_update["backfill_complete"] = True
        elif oldest_id_seen is not None:
            state_update["backfill_before_id"] = oldest_id_seen
            state_update["backfill_complete"] = False
        else:
            state_update["backfill_before_id"] = backfill_before_id
            state_update["backfill_complete"] = backfill_complete_state
    _save_state(state_file, state_update)
    LOGGER.info("Fetch done downloaded=%s backfill_complete=%s", len(downloaded), backfill_complete)
    return {"downloaded": downloaded, "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download new images from a public Telegram channel via t.me/s.",
    )
    parser.add_argument("--channel", default="vopros_dna", help="Telegram channel name")
    parser.add_argument(
        "--out-dir",
        default="vopros_dna/photos",
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
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file base path, supports {date} (default: data/telegram_fetch.log)",
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
    page_sleep = _resolve_float(args.page_sleep, "TELEGRAM_PAGE_SLEEP", 1.0)
    download_sleep = _resolve_float(args.download_sleep, "TELEGRAM_DOWNLOAD_SLEEP", 0.3)
    log_file = args.log_file or os.getenv("TELEGRAM_LOG_FILE") or "data/telegram_fetch.log"
    _setup_logging(Path(log_file))
    manifest_file = (
        args.manifest_file
        or os.getenv("TELEGRAM_MANIFEST_FILE")
        or "data/telegram_manifest.json"
    )
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
    )
    downloaded = result["downloaded"]
    failed = result["failed"]
    LOGGER.info("Downloaded %s images to %s", len(downloaded), args.out_dir)

    now = datetime.now(timezone.utc)
    manifest_path = Path(manifest_file)
    manifest = _load_manifest(manifest_path)
    for item in downloaded:
        filename = Path(item["path"]).name
        manifest[filename] = {
            "message_id": item.get("message_id"),
            "message_datetime_utc": item.get("message_datetime_utc"),
        }
    if downloaded:
        _save_manifest(manifest_path, manifest)
        LOGGER.info("Updated manifest %s", manifest_path)

    stats = _download_stats_by_date(downloaded, failed, now)
    if stats:
        index_path = Path(index_file)
        index = _load_index(index_path)
        for date_str, stat in stats.items():
            entry = index.get(date_str, {})
            downloaded_count = int(entry.get("downloaded_count", 0)) + stat["downloaded"]
            failed_count = int(entry.get("download_failed_count", 0)) + stat["failed"]
            entry.update(
                {
                    "date": date_str,
                    "downloaded_count": downloaded_count,
                    "downloaded_new_count": stat["downloaded"],
                    "download_failed_count": failed_count,
                    "download_failed_new_count": stat["failed"],
                    "download_failed": failed_count > 0,
                    "download_success": downloaded_count > 0 and failed_count == 0,
                }
            )
            index[date_str] = entry
        _write_index(index_path, index)
        LOGGER.info("Updated index %s", index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
