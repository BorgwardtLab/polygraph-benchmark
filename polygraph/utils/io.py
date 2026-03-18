"""I/O helpers for reproducibility experiment result streaming."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional


def _json_default(obj: Any) -> Any:
    """Fallback serializer for ``json.dumps``.

    Handles datetime objects, numpy scalars (via ``.item()``),
    and Path objects. Everything else is cast to ``str``.
    """
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def append_jsonl_locked(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append a JSON row to a shared file using an exclusive lock."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(row)
    payload.setdefault("ts", dt.datetime.now(tz=dt.timezone.utc).isoformat())

    with open(out_path, mode="a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(payload, default=_json_default))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def maybe_append_reproducibility_jsonl(
    row: Mapping[str, Any],
    *,
    path: Optional[str] = None,
    env_var: str = "POLYGRAPH_ASYNC_RESULTS_FILE",
) -> None:
    """Append a reproducibility result row if an output path is configured."""
    target = path or os.environ.get(env_var)
    if not target:
        return
    append_jsonl_locked(target, row)


def maybe_append_jsonl(
    row: Mapping[str, Any],
    *,
    path: Optional[str] = None,
    env_var: str = "POLYGRAPH_ASYNC_RESULTS_FILE",
) -> None:
    """Backward-compatible alias for reproducibility JSONL streaming."""
    maybe_append_reproducibility_jsonl(row, path=path, env_var=env_var)
