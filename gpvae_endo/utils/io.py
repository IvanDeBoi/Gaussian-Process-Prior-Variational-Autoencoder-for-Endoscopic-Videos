"""Small file and console helpers: atomic writes, a stdout tee to a log file, time formatting."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Callable


def atomic_save(path, write: Callable, mode: str = "wb") -> Path:
    """Write `path` atomically: `write(f)` fills a temporary file in the same folder, which is flushed,
    fsync'ed and then moved over `path`. On any error the temporary file is removed and an existing `path`
    is left untouched (a job killed half-way never leaves a broken checkpoint behind)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": "\n"})) as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


class Tee:
    """Duplicate everything written to stdout into a log file (append mode)."""

    def __init__(self, filepath):
        self.file = open(filepath, "a", encoding="utf-8")
        self.stdout = sys.stdout  # the original stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def format_hms(seconds: float) -> str:
    """Seconds -> 'HH:MM:SS'."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def prepend_text_file(file_path, text: str) -> None:
    """Write `text` at the top of `file_path`, keeping what was there below it (newest results first)."""
    existing_text = ""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_text = f.read()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
        if existing_text and not text.endswith("\n"):
            f.write("\n")
        if existing_text:
            f.write("\n")
            f.write(existing_text)
