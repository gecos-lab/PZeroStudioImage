"""Safe, versioned JSON project sessions (no executable pickle payloads)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


SESSION_VERSION = 1


@dataclass(slots=True)
class TraceRecord:
    """One accepted trace stored in pixel coordinates."""

    points: list[tuple[float, float]]
    confidence: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Session:
    """Serializable application state with explicit schema versioning."""

    image_path: str | None = None
    traces: list[TraceRecord] = field(default_factory=list)
    detector: dict[str, Any] = field(default_factory=dict)
    trace_settings: dict[str, Any] = field(default_factory=dict)
    version: int = SESSION_VERSION


def save_session(path: str | Path, session: Session) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(session)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_session(path: str | Path) -> Session:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0))
    if version != SESSION_VERSION:
        raise ValueError(f"Unsupported session version {version}; expected {SESSION_VERSION}.")
    traces = [TraceRecord(**record) for record in payload.get("traces", [])]
    return Session(
        image_path=payload.get("image_path"),
        traces=traces,
        detector=dict(payload.get("detector", {})),
        trace_settings=dict(payload.get("trace_settings", {})),
        version=version,
    )
