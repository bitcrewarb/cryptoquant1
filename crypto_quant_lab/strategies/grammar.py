"""Formal composable strategy grammar (boolean + temporal WITHIN/SEQUENCE + state-aware zones + structural execution), AST canonicalization, and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from crypto_quant_lab.core.exceptions import StrategyGrammarError

STRATEGY_FAMILIES: tuple[str, ...] = (
    "trend_following",
    "mean_reversion",
    "momentum_breakout",
    "volatility_breakout",
    "statistical_arbitrage",
    "pairs_trading",
    "pullback_trading",
    "range_trading",
    "liquidity_sweep",
    "bos_choch",
    "fvg_strategy",
    "order_block",
    "multi_timeframe",
    "hybrid_quant",
)

DIRECTIONS: tuple[str, ...] = ("long_only", "short_only", "long_short")
ENTRY_MODES: tuple[str, ...] = ("market", "zone_edge", "zone_50", "ote_62", "ote_70")
SL_MODES: tuple[str, ...] = ("atr", "sweep_wick", "zone_invalidation")
TP_MODES: tuple[str, ...] = ("atr", "rr_multiple", "opposing_liquidity")


@dataclass(frozen=True)
class LeafCondition:
    """Atomic mathematical or state-aware trading condition."""

    category: str       # trend | momentum | volatility | volume | statistical | price_action | smc | ict | zone_state | regime | mtf
    signal: str         # e.g. ema_cross_above, touching_fresh_zone, zone_age_lt
    params: dict[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "type": "leaf",
            "category": self.category,
            "signal": self.signal,
            "params": {k: self.params[k] for k in sorted(self.params)},
        }


@dataclass(frozen=True)
class LogicalExpr:
    """Boolean AST node supporting AND, OR, NOT."""

    op: str             # AND | OR | NOT
    children: tuple[Any, ...]

    def canonical_dict(self) -> dict[str, Any]:
        child_dicts = [c.canonical_dict() for c in self.children]
        if self.op in ("AND", "OR"):
            child_dicts.sort(key=lambda d: json.dumps(d, sort_keys=True))
        return {
            "type": "logic",
            "op": self.op,
            "children": child_dicts,
        }


@dataclass(frozen=True)
class TemporalExpr:
    """
    Causal temporal AST node:
    - WITHIN: True if child[0] was True within the last `window_bars` bars (inclusive of bar i).
    - SEQUENCE: True at bar i if children[0], children[1], ..., children[-1] fired in strict chronological order
      (`b_0 < b_1 < ... < b_{k-1} == i`) with max `window_bars` bars between consecutive steps.
    """

    op: str             # WITHIN | SEQUENCE
    children: tuple[Any, ...]
    window_bars: int = 5

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "type": "temporal",
            "op": self.op,
            "window_bars": int(self.window_bars),
            "children": [c.canonical_dict() for c in self.children],
        }


ConditionNode = LeafCondition | LogicalExpr | TemporalExpr


def node_from_dict(data: dict[str, Any]) -> ConditionNode:
    ntype = data.get("type")
    if ntype == "leaf":
        return LeafCondition(
            category=data["category"],
            signal=data["signal"],
            params=dict(data.get("params", {})),
        )
    if ntype == "logic":
        return LogicalExpr(
            op=data["op"],
            children=tuple(node_from_dict(c) for c in data.get("children", [])),
        )
    if ntype == "temporal":
        return TemporalExpr(
            op=data["op"],
            children=tuple(node_from_dict(c) for c in data.get("children", [])),
            window_bars=int(data.get("window_bars", 5)),
        )
    raise StrategyGrammarError(f"Invalid AST node type: {data}")


@dataclass
class StrategySpec:
    """Complete formal specification of a generated or user-defined quantitative strategy."""

    name: str
    family: str
    direction: str                  # long_only | short_only | long_short
    long_entry: ConditionNode | None
    long_exit: ConditionNode | None
    short_entry: ConditionNode | None
    short_exit: ConditionNode | None
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    atr_period: int = 14
    max_holding_bars: int | None = None
    trailing_stop_atr: float | None = None
    risk_per_trade_pct: float = 1.0
    version: int = 1
    # Structural entry & exit configuration (Phase 3)
    entry_mode: str = "market"              # market | zone_edge | zone_50 | ote_62 | ote_70
    entry_ttl_bars: int = 5
    sl_mode: str = "atr"                    # atr | sweep_wick | zone_invalidation
    sl_atr_buffer: float = 0.25
    tp_mode: str = "atr"                    # atr | rr_multiple | opposing_liquidity
    tp_rr_mult: float = 2.5
    partial_tp_ratio: float = 0.0           # fraction in [0, 1) to close at TP1
    partial_tp_rr: float = 1.5
    move_to_be_after_tp1: bool = False
    session_stop: bool = False
    template_id: str | None = None

    def to_rule_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "family": self.family,
            "direction": self.direction,
            "long_entry": self.long_entry.canonical_dict() if self.long_entry else None,
            "long_exit": self.long_exit.canonical_dict() if self.long_exit else None,
            "short_entry": self.short_entry.canonical_dict() if self.short_entry else None,
            "short_exit": self.short_exit.canonical_dict() if self.short_exit else None,
        }
        if self.template_id:
            out["template_id"] = self.template_id
        return out

    def to_params_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sl_atr_mult": round(float(self.sl_atr_mult), 4),
            "tp_atr_mult": round(float(self.tp_atr_mult), 4),
            "atr_period": int(self.atr_period),
            "max_holding_bars": self.max_holding_bars,
            "trailing_stop_atr": round(float(self.trailing_stop_atr), 4) if self.trailing_stop_atr else None,
            "risk_per_trade_pct": round(float(self.risk_per_trade_pct), 4),
        }
        if (
            self.entry_mode != "market"
            or self.sl_mode != "atr"
            or self.tp_mode != "atr"
            or self.partial_tp_ratio > 0.0
            or self.move_to_be_after_tp1
            or self.session_stop
        ):
            d.update(
                {
                    "entry_mode": self.entry_mode,
                    "entry_ttl_bars": int(self.entry_ttl_bars),
                    "sl_mode": self.sl_mode,
                    "sl_atr_buffer": round(float(self.sl_atr_buffer), 4),
                    "tp_mode": self.tp_mode,
                    "tp_rr_mult": round(float(self.tp_rr_mult), 4),
                    "partial_tp_ratio": round(float(self.partial_tp_ratio), 4),
                    "partial_tp_rr": round(float(self.partial_tp_rr), 4),
                    "move_to_be_after_tp1": bool(self.move_to_be_after_tp1),
                    "session_stop": bool(self.session_stop),
                }
            )
        return d

    def ast_hash(self) -> str:
        payload = {
            "rules": self.to_rule_dict(),
            "params": self.to_params_dict(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def param_hash(self) -> str:
        raw = json.dumps(self.to_params_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def from_dicts(cls, name: str, rule_json: dict[str, Any], params_json: dict[str, Any], version: int = 1) -> "StrategySpec":
        return cls(
            name=name,
            family=rule_json["family"],
            direction=rule_json["direction"],
            long_entry=node_from_dict(rule_json["long_entry"]) if rule_json.get("long_entry") else None,
            long_exit=node_from_dict(rule_json["long_exit"]) if rule_json.get("long_exit") else None,
            short_entry=node_from_dict(rule_json["short_entry"]) if rule_json.get("short_entry") else None,
            short_exit=node_from_dict(rule_json["short_exit"]) if rule_json.get("short_exit") else None,
            sl_atr_mult=float(params_json.get("sl_atr_mult", 2.0)),
            tp_atr_mult=float(params_json.get("tp_atr_mult", 3.0)),
            atr_period=int(params_json.get("atr_period", 14)),
            max_holding_bars=params_json.get("max_holding_bars"),
            trailing_stop_atr=params_json.get("trailing_stop_atr"),
            risk_per_trade_pct=float(params_json.get("risk_per_trade_pct", 1.0)),
            version=version,
            entry_mode=str(params_json.get("entry_mode", "market")),
            entry_ttl_bars=int(params_json.get("entry_ttl_bars", 5)),
            sl_mode=str(params_json.get("sl_mode", "atr")),
            sl_atr_buffer=float(params_json.get("sl_atr_buffer", 0.25)),
            tp_mode=str(params_json.get("tp_mode", "atr")),
            tp_rr_mult=float(params_json.get("tp_rr_mult", 2.5)),
            partial_tp_ratio=float(params_json.get("partial_tp_ratio", 0.0)),
            partial_tp_rr=float(params_json.get("partial_tp_rr", 1.5)),
            move_to_be_after_tp1=bool(params_json.get("move_to_be_after_tp1", False)),
            session_stop=bool(params_json.get("session_stop", False)),
            template_id=rule_json.get("template_id"),
        )


def _validate_tree(node: ConditionNode | None) -> None:
    if node is None:
        return
    if isinstance(node, LeafCondition):
        return
    if isinstance(node, LogicalExpr):
        if node.op not in ("AND", "OR", "NOT"):
            raise StrategyGrammarError(f"Unsupported LogicalExpr op: {node.op}")
        if node.op == "NOT" and len(node.children) != 1:
            raise StrategyGrammarError("NOT expression requires exactly 1 child")
        if node.op in ("AND", "OR") and len(node.children) < 1:
            raise StrategyGrammarError(f"{node.op} expression requires at least 1 child")
        for ch in node.children:
            _validate_tree(ch)
        return
    if isinstance(node, TemporalExpr):
        if node.op not in ("WITHIN", "SEQUENCE"):
            raise StrategyGrammarError(f"Unsupported TemporalExpr op: {node.op}")
        if node.window_bars < 1:
            raise StrategyGrammarError(f"Temporal window_bars must be >= 1, got {node.window_bars}")
        if node.op == "WITHIN" and len(node.children) != 1:
            raise StrategyGrammarError("WITHIN expression requires exactly 1 child")
        if node.op == "SEQUENCE" and len(node.children) < 2:
            raise StrategyGrammarError("SEQUENCE expression requires at least 2 ordered step children")
        for ch in node.children:
            _validate_tree(ch)
        return
    raise StrategyGrammarError(f"Unknown AST node: {node}")


def _collect_and_leaves(node: ConditionNode | None) -> list[LeafCondition]:
    """Collect conjunctive leaves under top-level AND for contradiction checking."""
    if node is None:
        return []
    if isinstance(node, LeafCondition):
        return [node]
    if isinstance(node, LogicalExpr) and node.op == "AND":
        out: list[LeafCondition] = []
        for ch in node.children:
            out.extend(_collect_and_leaves(ch))
        return out
    return []


def validate_strategy_spec(spec: StrategySpec, market_type: str | None = None) -> None:
    """Reject logically invalid, market-disallowed, or contradictory strategy specifications."""
    if spec.family not in STRATEGY_FAMILIES:
        raise StrategyGrammarError(f"Unknown strategy family: {spec.family}")
    if spec.direction not in DIRECTIONS:
        raise StrategyGrammarError(f"Invalid direction: {spec.direction}")
    if market_type == "spot" and spec.direction in ("short_only", "long_short"):
        raise StrategyGrammarError(f"Direction '{spec.direction}' is not allowed on spot market_type='spot'")
    if spec.entry_mode not in ENTRY_MODES:
        raise StrategyGrammarError(f"Unknown entry_mode: {spec.entry_mode}")
    if spec.sl_mode not in SL_MODES:
        raise StrategyGrammarError(f"Unknown sl_mode: {spec.sl_mode}")
    if spec.tp_mode not in TP_MODES:
        raise StrategyGrammarError(f"Unknown tp_mode: {spec.tp_mode}")
    if not (0.0 <= spec.partial_tp_ratio < 1.0):
        raise StrategyGrammarError("partial_tp_ratio must be in [0, 1)")
    if spec.sl_atr_mult <= 0 or spec.tp_atr_mult <= 0:
        raise StrategyGrammarError("Stop loss and take profit multipliers must be positive")
    if spec.risk_per_trade_pct <= 0 or spec.risk_per_trade_pct > 10.0:
        raise StrategyGrammarError("Risk per trade must be in (0, 10%]")
    if spec.direction in ("long_only", "long_short") and spec.long_entry is None:
        raise StrategyGrammarError("Long entry condition required")
    if spec.direction in ("short_only", "long_short") and spec.short_entry is None:
        raise StrategyGrammarError("Short entry condition required")

    for node in (spec.long_entry, spec.long_exit, spec.short_entry, spec.short_exit):
        _validate_tree(node)

    for side_node in (spec.long_entry, spec.short_entry):
        leaves = _collect_and_leaves(side_node)
        seen_sigs: set[str] = set()
        rsi_lowers: list[float] = []
        rsi_uppers: list[float] = []
        ema_pairs: dict[tuple[int, int], set[str]] = {}
        z_lowers: list[float] = []
        z_uppers: list[float] = []
        regime_sets: list[set[str]] = []

        for leaf in leaves:
            sig_key = json.dumps(leaf.canonical_dict(), sort_keys=True)
            if sig_key in seen_sigs:
                raise StrategyGrammarError("Duplicate identical condition inside AND clause")
            seen_sigs.add(sig_key)

            if leaf.signal == "rsi_below":
                rsi_uppers.append(float(leaf.params.get("threshold", 30)))
            elif leaf.signal == "rsi_above":
                rsi_lowers.append(float(leaf.params.get("threshold", 70)))
            elif leaf.signal in ("ema_gt_ema", "ema_lt_ema"):
                f = int(leaf.params.get("fast", 10))
                s = int(leaf.params.get("slow", 50))
                if f >= s:
                    raise StrategyGrammarError(f"Fast EMA ({f}) must be < Slow EMA ({s})")
                ema_pairs.setdefault((f, s), set()).add(leaf.signal)
            elif leaf.signal == "zscore_below":
                z_uppers.append(float(leaf.params.get("threshold", -1.5)))
            elif leaf.signal == "zscore_above":
                z_lowers.append(float(leaf.params.get("threshold", 1.5)))
            elif leaf.signal == "regime_in":
                regime_sets.append(set(leaf.params.get("regimes", [])))

        if rsi_lowers and rsi_uppers and max(rsi_lowers) >= min(rsi_uppers):
            raise StrategyGrammarError("Contradictory RSI bounds in entry rule")
        if z_lowers and z_uppers and max(z_lowers) >= min(z_uppers):
            raise StrategyGrammarError("Contradictory Z-score bounds in entry rule")
        for pair, ops in ema_pairs.items():
            if len(ops) > 1:
                raise StrategyGrammarError(f"Contradictory EMA directions for {pair}")
        if len(regime_sets) > 1:
            inter = set.intersection(*regime_sets)
            if not inter:
                raise StrategyGrammarError("Disjoint market regime filters in AND clause")
