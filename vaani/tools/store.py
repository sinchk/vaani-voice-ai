"""Tiny JSON-file store with copy-on-first-use seeding.

Tracked files under ``data/`` are *seed* data. Mutations are written to a copy
in the runtime state directory so that using the app never dirties the git
working tree, and ``reset()`` restores the demo data.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any


class JsonStore:
    def __init__(self, seed_path: Path, state_path: Path) -> None:
        self.seed_path = Path(seed_path)
        self.state_path = Path(state_path)
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        """Hold this around a read-modify-write sequence."""
        return self._lock

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.state_path.exists():
                self.reset()
            with self.state_path.open(encoding="utf-8") as fh:
                return json.load(fh)

    def save(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self.state_path)  # atomic: never leaves a half-written file

    def reset(self) -> None:
        with self._lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.seed_path, self.state_path)
