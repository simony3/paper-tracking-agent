"""Small, safe helpers for the project's JSON state files.

Runtime state is intentionally local, but writes should still be atomic: an
interrupted cron run must not leave half-written JSON behind.  ``update_json``
also uses a best-effort advisory lock on Unix so two overlapping runs do not
silently overwrite each other's read-modify-write updates.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator

try:  # ``fcntl`` is unavailable on Windows; atomic replacement still applies.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None


class DataFileError(RuntimeError):
    """Raised when a local state file exists but is not valid JSON."""


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataFileError(f"状态文件无法读取: {path} ({exc})") from exc


def atomic_write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=indent)
    atomic_write_text(path, text)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@contextmanager
def json_file_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def update_json(path: Path, default: Any, updater: Callable[[Any], Any]) -> Any:
    """Atomically apply ``updater`` to the latest on-disk JSON value."""
    with json_file_lock(path):
        current = read_json(path, default)
        updated = updater(current)
        atomic_write_json(path, updated)
        return updated
