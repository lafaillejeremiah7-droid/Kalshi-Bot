from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_CONTROL_URL = (
    "https://raw.githubusercontent.com/lafaillejeremiah7-droid/"
    "XAUUSD-Company/main/runtime-control.json"
)


def parse_enabled(payload: str | bytes | dict[str, Any]) -> bool:
    if isinstance(payload, (str, bytes)):
        data = json.loads(payload)
    else:
        data = payload
    if not isinstance(data, dict) or not isinstance(data.get("enabled"), bool):
        raise ValueError("runtime control must contain boolean 'enabled'")
    return bool(data["enabled"])


def fetch_enabled(url: str, timeout: float = 10.0, session: Any = requests) -> bool:
    separator = "&" if "?" in url else "?"
    cache_busted = f"{url}{separator}_ts={time.time_ns()}"
    response = session.get(
        cache_busted,
        timeout=timeout,
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )
    response.raise_for_status()
    return parse_enabled(response.text)


def terminate_process(process: subprocess.Popen[Any], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@dataclass(frozen=True)
class RuntimeControlConfig:
    url: str = os.getenv("RUNTIME_CONTROL_URL", DEFAULT_CONTROL_URL)
    poll_seconds: int = int(os.getenv("RUNTIME_CONTROL_POLL_SECONDS", "15"))
    request_timeout_seconds: float = float(os.getenv("RUNTIME_CONTROL_TIMEOUT_SECONDS", "10"))
    max_failures: int = int(os.getenv("RUNTIME_CONTROL_MAX_FAILURES", "2"))
    stop_grace_seconds: float = float(os.getenv("RUNTIME_STOP_GRACE_SECONDS", "10"))

    def validate(self) -> None:
        if self.poll_seconds < 5:
            raise ValueError("RUNTIME_CONTROL_POLL_SECONDS must be at least 5")
        if self.request_timeout_seconds <= 0:
            raise ValueError("RUNTIME_CONTROL_TIMEOUT_SECONDS must be positive")
        if not 1 <= self.max_failures <= 5:
            raise ValueError("RUNTIME_CONTROL_MAX_FAILURES must be between 1 and 5")
        if self.stop_grace_seconds <= 0:
            raise ValueError("RUNTIME_STOP_GRACE_SECONDS must be positive")


def run() -> int:
    cfg = RuntimeControlConfig()
    cfg.validate()

    try:
        enabled = fetch_enabled(cfg.url, cfg.request_timeout_seconds)
    except Exception as exc:
        print(f"Runtime control unavailable at startup; refusing to start: {exc}", flush=True)
        return 2

    if not enabled:
        print("Runtime control is OFF; company will not start.", flush=True)
        return 0

    print("Runtime control is ON; starting XAU/USD company.", flush=True)
    process = subprocess.Popen([sys.executable, "main.py"])
    failures = 0

    try:
        while True:
            code = process.poll()
            if code is not None:
                print(f"Company process exited with code {code}.", flush=True)
                return int(code)

            time.sleep(cfg.poll_seconds)
            try:
                enabled = fetch_enabled(cfg.url, cfg.request_timeout_seconds)
            except Exception as exc:
                failures += 1
                print(
                    f"Runtime control check failed ({failures}/{cfg.max_failures}): {exc}",
                    flush=True,
                )
                if failures >= cfg.max_failures:
                    print("Control cannot be verified; fail-closing company.", flush=True)
                    terminate_process(process, cfg.stop_grace_seconds)
                    return 3
                continue

            failures = 0
            if not enabled:
                print("Runtime control switched OFF; stopping company cleanly.", flush=True)
                terminate_process(process, cfg.stop_grace_seconds)
                return 0
    except KeyboardInterrupt:
        print("Runtime interrupted; stopping company cleanly.", flush=True)
        terminate_process(process, cfg.stop_grace_seconds)
        return 130


if __name__ == "__main__":
    raise SystemExit(run())
