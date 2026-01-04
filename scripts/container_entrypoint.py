import os
import subprocess
import sys
import time


def _resolve_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_int(raw: str | None, default: int, min_value: int = 1) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _run_pipeline() -> int:
    steps = os.getenv("PIPELINE_STEPS", "all")
    env_file = os.getenv("PIPELINE_ENV_FILE")
    cmd = [sys.executable, "-m", "scripts.pipeline_steps", steps]
    if env_file:
        cmd.extend(["--env-file", env_file])
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    run_on_start = _resolve_bool(os.getenv("PIPELINE_RUN_ON_START"), False)
    keepalive = _resolve_bool(os.getenv("PIPELINE_KEEPALIVE"), True)
    idle_sleep = _resolve_int(os.getenv("PIPELINE_IDLE_SLEEP"), 3600)

    if run_on_start:
        exit_code = _run_pipeline()
        if exit_code != 0:
            return exit_code

    if keepalive:
        while True:
            time.sleep(idle_sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
