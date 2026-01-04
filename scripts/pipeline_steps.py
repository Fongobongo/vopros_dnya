import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"


def _run_command(cmd: list[str]) -> int:
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _script_path(name: str) -> str:
    return str(SCRIPTS_DIR / name)


def _run_fetch(env_file: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, _script_path("fetch_channel_images.py"), "--env-file", env_file]
    cmd.extend(extra_args)
    return _run_command(cmd)


def _run_preprocess(env_file: str, input_dir: str, extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        _script_path("preprocess_questions.py"),
        input_dir,
        "--env-file",
        env_file,
    ]
    cmd.extend(extra_args)
    return _run_command(cmd)


def _run_ocr(env_file: str, input_dir: str, extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        _script_path("ocr_images.py"),
        input_dir,
        "--env-file",
        env_file,
    ]
    cmd.extend(extra_args)
    return _run_command(cmd)


def _run_moderate(env_file: str, input_dir: str, pattern: str, extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        _script_path("moderate_json.py"),
        "--input-dir",
        input_dir,
        "--pattern",
        pattern,
        "--env-file",
        env_file,
    ]
    cmd.extend(extra_args)
    return _run_command(cmd)


def _run_export(env_file: str, input_dir: str, pattern: str, extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        _script_path("export_validated_to_sqlite.py"),
        "--input-dir",
        input_dir,
        "--pattern",
        pattern,
        "--env-file",
        env_file,
    ]
    cmd.extend(extra_args)
    return _run_command(cmd)


def _run_all(env_file: str, extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        _script_path("daily_pipeline.py"),
        "--env-file",
        env_file,
    ]
    cmd.extend(extra_args)
    return _run_command(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pipeline steps (fetch, preprocess, ocr, moderate, export).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch new images")
    fetch_parser.add_argument("--env-file", default=".env")
    fetch_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    preprocess_parser = subparsers.add_parser("preprocess", help="Crop and filter images")
    preprocess_parser.add_argument("--env-file", default=".env")
    preprocess_parser.add_argument("--input-dir", default="data/photos")
    preprocess_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    ocr_parser = subparsers.add_parser("ocr", help="Run OCR on cropped images")
    ocr_parser.add_argument("--env-file", default=".env")
    ocr_parser.add_argument("--input-dir", default="data/cropped")
    ocr_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    moderate_parser = subparsers.add_parser("moderate", help="Run Mistral moderation")
    moderate_parser.add_argument("--env-file", default=".env")
    moderate_parser.add_argument("--input-dir", default="data")
    moderate_parser.add_argument("--pattern", default="questions_*.json")
    moderate_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    export_parser = subparsers.add_parser("export", help="Export validated phrases to SQLite")
    export_parser.add_argument("--env-file", default=".env")
    export_parser.add_argument("--input-dir", default="data")
    export_parser.add_argument("--pattern", default="questions_*.json")
    export_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    all_parser = subparsers.add_parser("all", help="Run full daily pipeline")
    all_parser.add_argument("--env-file", default=".env")
    all_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == "fetch":
        return _run_fetch(args.env_file, args.extra_args)
    if args.command == "preprocess":
        return _run_preprocess(args.env_file, args.input_dir, args.extra_args)
    if args.command == "ocr":
        return _run_ocr(args.env_file, args.input_dir, args.extra_args)
    if args.command == "moderate":
        return _run_moderate(args.env_file, args.input_dir, args.pattern, args.extra_args)
    if args.command == "export":
        return _run_export(args.env_file, args.input_dir, args.pattern, args.extra_args)
    if args.command == "all":
        return _run_all(args.env_file, args.extra_args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
