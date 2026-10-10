"""Managed local solver subprocess. No network service or hidden fallback."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[1]


class SolverError(RuntimeError):
    pass


def select_binary(root=ROOT):
    """Resolve one immutable release, with an explicit experimental override."""
    override=os.environ.get("HU_SOLVER_BINARY")
    if override:return Path(override).expanduser().resolve()
    manifest=root/"native/runtime.json"
    if not manifest.exists():return (root/"native/target/release/hu-solver").resolve()
    try:
        record=json.loads(manifest.read_text());binary=(root/record["binary"]).resolve()
        if not binary.is_relative_to(root.resolve()) or not binary.is_file():raise ValueError("Missing or invalid release path")
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=record["sha256"]:raise ValueError("Release hash mismatch")
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SolverError(f"Invalid native runtime manifest; rebuild with training/build_native.py: {exc}") from exc
    return binary


# Resolving once at import pins all future child processes in this Python
# policy to the same executable, even if a later build publishes a new release.
DEFAULT_BINARY=select_binary()


class NativeSolver:
    def __init__(self, threads=6):
        binary = DEFAULT_BINARY
        if not binary.exists():
            raise SolverError("Build the solver first: .venv/bin/python training/build_native.py")
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            env={**os.environ, "HU_SOLVER_THREADS": str(threads)},
        )
        self.lock = threading.Lock()
        atexit.register(self.close)

    def request(self, payload, timeout=180):
        with self.lock:
            if self.process.poll() is not None:
                raise SolverError("Native solver exited")
            self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not ready:
                self.close()
                raise SolverError(f"Native solver exceeded {timeout:g}s hard timeout")
            line = self.process.stdout.readline()
            if not line:
                raise SolverError("Native solver closed its response stream")
            result = json.loads(line)
            if "error" in result:
                raise SolverError(result["error"])
            return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe and not pipe.closed:
                pipe.close()
        atexit.unregister(self.close)
