from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.session_id = f"{time.time_ns()}-{os.getpid()}"
        self.path = directory / f"rebot_rs_{time.strftime('%Y%m%d_%H%M%S')}_{self.session_id}.jsonl"
        self._file = self.path.open("x", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> None:
        record = {"time": time.time(), "monotonic": time.monotonic(),
                  "session_id": self.session_id, "event": event, **fields}
        self._file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
