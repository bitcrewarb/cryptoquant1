"""Unified causal Zone, Event, and DetectionResult domain objects shared by strategies, signal engine, and charts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ZoneObject:
    """Structured price zone / level with full causal lifecycle tracking."""

    id: str
    type: str               # fvg | ifvg | ob | breaker | bpr | volume_imbalance | liquidity_pool | sr | ote | nwog | ndog
    direction: str          # bull | bear | neutral
    top: float
    bottom: float
    created_bar: int
    created_time: int       # ms epoch
    confirmed_time: int     # ms epoch when causally confirmed
    status: str = "fresh"   # fresh | tested | mitigated | invalidated | expired
    mitigated_time: int | None = None
    expiry: int | None = None       # bar index expiry
    source_event: str | None = None
    strength: float = 1.0
    touches: int = 0
    fill_pct: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) * 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "direction": self.direction,
            "top": round(float(self.top), 8),
            "bottom": round(float(self.bottom), 8),
            "midpoint": round(self.midpoint, 8),
            "created_bar": int(self.created_bar),
            "created_time": int(self.created_time),
            "confirmed_time": int(self.confirmed_time),
            "status": self.status,
            "mitigated_time": int(self.mitigated_time) if self.mitigated_time is not None else None,
            "expiry": int(self.expiry) if self.expiry is not None else None,
            "source_event": self.source_event,
            "strength": round(float(self.strength), 4),
            "touches": int(self.touches),
            "fill_pct": round(float(self.fill_pct), 4),
            "meta": self.meta,
        }


@dataclass
class EventObject:
    """Structured market event (BOS, CHoCH, MSS, Liquidity Sweep, Flip, Candlestick/Chart Pattern)."""

    id: str
    type: str
    direction: str          # bull | bear | neutral
    time: int               # ms epoch
    bar_index: int
    price: float
    level: float | None = None
    linked_zone_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "direction": self.direction,
            "time": int(self.time),
            "bar_index": int(self.bar_index),
            "price": round(float(self.price), 8),
            "level": round(float(self.level), 8) if self.level is not None and np.isfinite(self.level) else None,
            "linked_zone_ids": list(self.linked_zone_ids),
            "meta": self.meta,
        }


@dataclass
class DetectionResult:
    """Single source of truth returned by causal detectors (arrays + structured zones, events, and levels)."""

    arrays: dict[str, np.ndarray]
    zones: list[ZoneObject] = field(default_factory=list)
    events: list[EventObject] = field(default_factory=list)
    levels: dict[str, Any] = field(default_factory=dict)

    def active_zones(self, zone_type: str | None = None, statuses: tuple[str, ...] = ("fresh", "tested")) -> list[ZoneObject]:
        return [
            z for z in self.zones
            if z.status in statuses and (zone_type is None or z.type == zone_type)
        ]
