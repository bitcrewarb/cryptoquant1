"""Realistic transaction cost, volatility/size slippage, spread, margin liquidation, and futures funding models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionCostConfig:
    market_type: str = "spot"          # spot | futures
    maker_fee_bps: float = 10.0        # Binance spot standard 0.10% = 10 bps; futures 2 bps
    taker_fee_bps: float = 10.0        # Binance spot taker 10 bps; futures 5 bps
    slippage_bps: float = 3.0          # Base market impact / slippage in bps
    spread_bps: float = 2.0            # Half-spread crossing cost in bps (total spread = 2 bps -> 1 bps per side)
    funding_rate_per_8h: float = 0.0001  # 0.01% per 8h default when historical funding series not attached
    leverage: float = 1.0
    max_volume_participation: float = 0.10  # Max 10% of bar volume per fill (triggers partial fills if exceeded)
    vol_slippage_scale: float = 0.0         # Optional dynamic ATR% slippage multiplier (bps per 1% bar ATR)
    size_impact_bps_per_vol_pct: float = 0.0  # Optional quadratic-like impact bps per 1% of bar volume consumed
    maintenance_margin_rate: float = 0.005  # 0.5% futures maintenance margin rate for liquidation check

    def _impact_rate(
        self,
        raw_price: float,
        atr_val: float | None = None,
        order_qty: float | None = None,
        bar_volume: float | None = None,
    ) -> float:
        base_bps = self.slippage_bps + self.spread_bps * 0.5
        if self.vol_slippage_scale > 0.0 and atr_val is not None and np.isfinite(atr_val) and raw_price > 0:
            atr_pct = (atr_val / raw_price) * 100.0
            base_bps += atr_pct * self.vol_slippage_scale
        if (
            self.size_impact_bps_per_vol_pct > 0.0
            and order_qty is not None
            and bar_volume is not None
            and bar_volume > 0
        ):
            part_pct = min(100.0, (abs(order_qty) / bar_volume) * 100.0)
            base_bps += part_pct * self.size_impact_bps_per_vol_pct
        return max(0.0, base_bps) / 10_000.0

    def effective_entry_price(
        self,
        raw_price: float,
        side: str,
        is_maker: bool = False,
        atr_val: float | None = None,
        order_qty: float | None = None,
        bar_volume: float | None = None,
    ) -> tuple[float, float]:
        """Return (fill_price, slippage_cost_per_unit)."""
        if is_maker:
            return raw_price, 0.0
        impact_rate = self._impact_rate(raw_price, atr_val=atr_val, order_qty=order_qty, bar_volume=bar_volume)
        if side == "LONG":
            fill = raw_price * (1.0 + impact_rate)
            return fill, fill - raw_price
        fill = raw_price * (1.0 - impact_rate)
        return fill, raw_price - fill

    def effective_exit_price(
        self,
        raw_price: float,
        side: str,
        is_maker: bool = False,
        atr_val: float | None = None,
        order_qty: float | None = None,
        bar_volume: float | None = None,
    ) -> tuple[float, float]:
        """Return (fill_price, slippage_cost_per_unit)."""
        if is_maker:
            return raw_price, 0.0
        impact_rate = self._impact_rate(raw_price, atr_val=atr_val, order_qty=order_qty, bar_volume=bar_volume)
        if side == "LONG":
            fill = raw_price * (1.0 - impact_rate)
            return fill, raw_price - fill
        fill = raw_price * (1.0 + impact_rate)
        return fill, fill - raw_price

    def commission(self, notional: float, is_maker: bool = False) -> float:
        bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        return abs(notional) * (bps / 10_000.0)

    def liquidation_price(self, entry_price: float, side: str) -> float | None:
        """Isolated margin liquidation price for leveraged futures positions."""
        if self.market_type != "futures" or self.leverage <= 1.0 or entry_price <= 0:
            return None
        margin_buffer = max(0.0, (1.0 / float(self.leverage)) - float(self.maintenance_margin_rate))
        if side == "LONG":
            return entry_price * max(0.0, 1.0 - margin_buffer)
        return entry_price * (1.0 + margin_buffer)
