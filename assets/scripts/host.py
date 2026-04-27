from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

PROTOCOL_VERSION = 1
HERE = Path(__file__).resolve().parent

STDIN = sys.stdin.buffer
STDOUT = sys.stdout.buffer
STDOUT_LOCK = threading.Lock()


def log(msg: str) -> None:
    sys.stderr.write(f"[py_host] {msg}\n")
    sys.stderr.flush()


def send_ctrl(obj: dict, payload: bytes | None = None) -> None:
    line = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    with STDOUT_LOCK:
        STDOUT.write(line)
        if payload is not None:
            STDOUT.write(payload)
        STDOUT.flush()


def emit_status(phase: str, progress: float = 0.0, message: str = "") -> None:
    p = float(max(0.0, min(1.0, progress)))
    send_ctrl({"type": "status", "phase": str(phase),
               "progress": p, "msg": str(message)})


def read_exact(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = STDIN.read(n - len(buf))
        if not chunk:
            return bytes(buf)
        buf.extend(chunk)
    return bytes(buf)


def read_line() -> bytes:
    buf = bytearray()
    while True:
        b = STDIN.read(1)
        if not b:
            return bytes(buf)
        if b == b"\n":
            return bytes(buf)
        if b == b"\r":
            continue
        buf.extend(b)


def discover_effects() -> list[str]:
    out = []
    for p in sorted(HERE.glob("*.py")):
        if p.name == "host.py" or p.name.startswith("_"):
            continue
        out.append(p.stem)
    return out


def import_effect(name: str):
    path = HERE / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"effect not found: {path}")
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(f"fox_effect_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"spec failed for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for fn in ("load", "apply", "unload"):
        if not hasattr(module, fn):
            raise AttributeError(f"{name}.py is missing required '{fn}' function")
    return module

_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _requirement_installed(spec: str) -> bool:
    m = _REQ_NAME_RE.match(spec)
    if not m:
        return False
    try:
        importlib.metadata.version(m.group(1))
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def ensure_requires(name: str, reqs) -> None:
    if not reqs:
        return
    missing = [r for r in reqs if not _requirement_installed(r)]
    if not missing:
        return
    pretty = " ".join(missing)
    emit_status("installing", 0.0, f"pip install {pretty}")
    log(f"ensuring deps for {name}: {missing}")
    cmd = [sys.executable, "-m", "pip", "install", *missing]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = proc.stdout.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"pip install {pretty} failed (exit {proc.returncode}):\n{tail}")
    importlib.invalidate_caches()
    emit_status("installing", 1.0, f"installed {pretty}")


class Host:
    def __init__(self) -> None:
        self.active_name: str | None = None
        self.active_module = None
        self.active_info: dict = {}
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.busy = threading.Event()

    def do_load(self, name: str) -> None:
        if self.active_name == name and self.active_module is not None:
            send_ctrl({"type": "loaded", "name": name,
                       "label": self.active_info.get("name", name),
                       "cached": True})
            return
        self.do_unload()
        try:
            mod = import_effect(name)
            ensure_requires(name, getattr(mod, "REQUIRES", None))
            info = mod.load() or {}
            self.active_name = name
            self.active_module = mod
            self.active_info = info
            send_ctrl({"type": "loaded", "name": name,
                       "label": info.get("name", name),
                       "cached": False})
            log(f"loaded effect '{name}'")
        except Exception as ex:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            send_ctrl({"type": "load_error", "name": name, "msg": str(ex)})

    def do_unload(self) -> None:
        if self.active_module is None:
            return
        try:
            self.active_module.unload()
        except Exception as ex:
            log(f"unload() of {self.active_name} raised: {ex}")
        self.active_module = None
        self.active_name = None
        self.active_info = {}
    def do_infer(self, w: int, h: int, rgb: bytes) -> None:
        if self.active_module is None:
            send_ctrl({"type": "error",
                       "msg": "no effect loaded (send load first)"})
            return
        if self.busy.is_set():
            send_ctrl({"type": "error", "msg": "inference already running"})
            return
        self.cancel_event.clear()
        self.busy.set()

        def worker():
            try:
                def emit(partial: bytes, progress: float) -> None:
                    if len(partial) != w * h * 3:
                        raise ValueError(
                            f"emit_partial: expected {w*h*3} bytes, "
                            f"got {len(partial)}")
                    send_ctrl(
                        {"type": "partial",
                         "progress": float(max(0.0, min(1.0, progress))),
                         "bytes": len(partial)},
                        partial)

                def should_cancel() -> bool:
                    return self.cancel_event.is_set()

                out = self.active_module.apply(rgb, w, h, emit, should_cancel)
                if self.cancel_event.is_set():
                    send_ctrl({"type": "cancelled"})
                else:
                    if not isinstance(out, (bytes, bytearray, memoryview)):
                        raise TypeError(
                            f"apply() must return bytes, got {type(out).__name__}")
                    payload = bytes(out)
                    if len(payload) != w * h * 3:
                        raise ValueError(
                            f"apply(): final frame must be {w*h*3} bytes, "
                            f"got {len(payload)}")
                    send_ctrl({"type": "done", "bytes": len(payload)}, payload)
            except Exception as ex:
                tb = traceback.format_exc()
                sys.stderr.write(tb)
                send_ctrl({"type": "error", "msg": str(ex)})
            finally:
                self.busy.clear()

        self.worker = threading.Thread(target=worker, daemon=True,
                                       name=f"effect-{self.active_name}")
        self.worker.start()

    def do_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> int:
        send_ctrl({"type": "hello", "pid": os.getpid(),
                   "version": PROTOCOL_VERSION})
        try:
            while True:
                line = read_line()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception as ex:
                    send_ctrl({"type": "error",
                               "msg": f"bad json: {ex}: {line!r}"})
                    continue
                op = msg.get("op")
                if op == "list":
                    send_ctrl({"type": "list", "names": discover_effects()})
                elif op == "load":
                    self.do_load(str(msg.get("name", "")))
                elif op == "unload":
                    self.do_unload()
                    send_ctrl({"type": "unloaded"})
                elif op == "infer":
                    w = int(msg.get("w", 0))
                    h = int(msg.get("h", 0))
                    n = int(msg.get("bytes", 0))
                    if w <= 0 or h <= 0 or n != w * h * 3:
                        send_ctrl({"type": "error",
                                   "msg": f"bad infer header: w={w} h={h} n={n}"})
                        if n > 0:
                            read_exact(n)
                        continue
                    rgb = read_exact(n)
                    if len(rgb) != n:
                        send_ctrl({"type": "error",
                                   "msg": "short read on infer payload"})
                        break
                    self.do_infer(w, h, rgb)
                elif op == "cancel":
                    self.do_cancel()
                elif op == "shutdown":
                    break
                else:
                    send_ctrl({"type": "error",
                               "msg": f"unknown op: {op!r}"})
        finally:
            self.do_unload()
            send_ctrl({"type": "bye"})
        return 0


if __name__ == "__main__":
    try:
        sys.exit(Host().run())
    except Exception:
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
