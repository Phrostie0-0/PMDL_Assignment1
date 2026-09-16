"""Wait for the deployed model API to become healthy."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def wait_for_api(url: str, timeout: int, interval: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("status") == "ok":
                print(f"API is healthy: {payload}")
                return
            last_error = f"unexpected response: {payload}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(interval)
    raise RuntimeError(f"API did not become healthy within {timeout}s: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.getenv("API_HEALTH_URL", "http://localhost:8000/health"),
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    wait_for_api(arguments.url, arguments.timeout, arguments.interval)

