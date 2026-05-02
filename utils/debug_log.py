"""Structured sidecar debug logging for interpreter runs.

The raw trace format is intentionally optimized for high-volume value traces.
This module is for low-volume explanatory events: branch choices, solver
queries, candidate decisions, and setup diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable


DEFAULT_DEBUG_CATEGORIES = frozenset({
    "exec",
    "input",
    "mem",
    "branch",
    "solver",
    "candidate",
    "unsupported",
    "trace",
})

DEBUG_SCHEMA_VERSION = 1

_MAX_STRING = 4096
_MAX_ITEMS = 64


def parse_debug_categories(value: str | Iterable[str] | None) -> frozenset[str]:
    if value is None:
        return DEFAULT_DEBUG_CATEGORIES
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    else:
        raw = [str(part).strip() for part in value]
    cats = {part for part in raw if part}
    if not cats or "all" in cats:
        return DEFAULT_DEBUG_CATEGORIES
    return frozenset(cats)


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_STRING:
            return value
        return value[:_MAX_STRING] + f"...<truncated {len(value) - _MAX_STRING} chars>"
    if isinstance(value, bytes):
        text = value[:_MAX_STRING].hex()
        if len(value) > _MAX_STRING:
            text += f"...<truncated {len(value) - _MAX_STRING} bytes>"
        return text
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        items = list(value.items())
        out = {
            str(k): _sanitize(v)
            for k, v in items[:_MAX_ITEMS]
        }
        if len(items) > _MAX_ITEMS:
            out["__truncated_items__"] = len(items) - _MAX_ITEMS
        return out
    if isinstance(value, (list, tuple)):
        out = [_sanitize(v) for v in list(value)[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            out.append({"__truncated_items__": len(value) - _MAX_ITEMS})
        return out
    if isinstance(value, set):
        items = sorted(str(v) for v in value)
        out = items[:_MAX_ITEMS]
        if len(items) > _MAX_ITEMS:
            out.append(f"<truncated {len(items) - _MAX_ITEMS} items>")
        return out
    text = repr(value)
    if len(text) > _MAX_STRING:
        text = text[:_MAX_STRING] + f"...<truncated {len(text) - _MAX_STRING} chars>"
    return text


@dataclass(frozen=True)
class DebugLogger:
    root: Path | None = None
    categories: frozenset[str] = DEFAULT_DEBUG_CATEGORIES
    run_id: str = "run"
    stream_name: str = "debug"
    bound_fields: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def disabled(cls) -> "DebugLogger":
        return cls(root=None)

    @classmethod
    def from_options(
        cls,
        root: str | Path | None,
        categories: str | Iterable[str] | None = None,
        *,
        run_id: str | None = None,
        stream_name: str = "debug",
    ) -> "DebugLogger":
        if not root:
            return cls.disabled()
        rid = run_id or time.strftime("%Y%m%d_%H%M%S")
        return cls(
            root=Path(root),
            categories=parse_debug_categories(categories),
            run_id=rid,
            stream_name=stream_name,
        )

    @property
    def enabled(self) -> bool:
        return self.root is not None

    @property
    def path(self) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{self.stream_name}.jsonl"

    def is_enabled(self, category: str) -> bool:
        return self.enabled and category in self.categories

    def bind(self, **fields: Any) -> "DebugLogger":
        if not fields:
            return self
        return replace(
            self,
            bound_fields=self.bound_fields + tuple(sorted(fields.items())),
        )

    def event(self, category: str, event: str, **fields: Any) -> None:
        if not self.is_enabled(category):
            return
        assert self.root is not None
        record = {
            "schema_version": DEBUG_SCHEMA_VERSION,
            "ts_ns": time.time_ns(),
            "pid": os.getpid(),
            "run_id": self.run_id,
            "category": category,
            "event": event,
        }
        for key, value in self.bound_fields:
            record[key] = _sanitize(value)
        for key, value in fields.items():
            record[key] = _sanitize(value)
        path = self.path
        assert path is not None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                fh.write("\n")
        except OSError:
            return

    def native_env(self) -> dict[str, str]:
        if not self.enabled or self.root is None:
            return {}
        return {
            "SWOOSH_DEBUG_LOG": str(self.path),
            "SWOOSH_DEBUG_CATEGORIES": ",".join(sorted(self.categories)),
            "SWOOSH_DEBUG_RUN_ID": self.run_id,
        }
