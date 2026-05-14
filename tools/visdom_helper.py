"""Helper to check and start a local visdom server non-blocking.

Usage:
    from tools.visdom_helper import start_visdom_server, is_visdom_running
    start_visdom_server(port=8097, host='127.0.0.1')
"""
from __future__ import annotations
import subprocess
import sys
import time
import urllib.request
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def is_visdom_running(host: str = "127.0.0.1", port: int = 8097, timeout: float = 1.0) -> bool:
    url = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_visdom_server(python_executable: str | None = None,
                        port: int = 8097,
                        host: str = "127.0.0.1",
                        detach: bool = True,
                        timeout: float = 10.0) -> bool:
    """Start a visdom server process if one is not already running.

    Returns True if a server is reachable after the call, False otherwise.
    """
    if is_visdom_running(host=host, port=port, timeout=1.0):
        logger.info("Visdom already running at %s:%s", host, port)
        return True

    py = python_executable or sys.executable
    cmd = [py, "-m", "visdom.server", "-p", str(port), "--hostname", host]

    env = os.environ.copy()
    # ensure log directory
    logdir = Path(os.path.expanduser("~")) / ".visdom_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    stdout = open(logdir / f"visdom_{port}.out", "a")
    stderr = open(logdir / f"visdom_{port}.err", "a")

    popen_kwargs = {}
    if os.name == "nt":
        # DETACHED_PROCESS + CREATE_NO_WINDOW makes it run in background without a console
        DETACHED_PROCESS = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        popen_kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True

    try:
        subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, **popen_kwargs)
    except Exception as exc:
        logger.error("Failed to spawn visdom server: %s", exc)
        return False

    # wait for server to come up
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_visdom_running(host=host, port=port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Start a local visdom server (helper)")
    p.add_argument("--port", type=int, default=8097)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--timeout", type=float, default=10.0)
    args = p.parse_args()
    ok = start_visdom_server(port=args.port, host=args.host, timeout=args.timeout)
    print("visdom running:" , ok)
