"""Execution-time Risk Manager enforcing sizing, exposure, drawdown, daily loss, and emergency halt rules."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_quant_lab.core.exceptions import RiskViolationError


@dataclass
class RiskConfig:
    sizing_mode: str = "atr_based"       # fixed_fractional | atr_based | volatility_based
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 5.0
    max_portfolio_drawdown_pct: float = 20.0
    max_open_positions: int = 10
    max_asset_exposure_pct: float = 25.0
    max_leverage: float = 3.0
    max_consecutive_losses: int = 8
    emergency_halt: bool = False


class RiskManager:
    """Real execution-time risk enforcement used by event-driven backtester, paper trading, and live router."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def calculate_position_size(
        self,
        equity: float,
        price: float,
        atr_value: float | None = None,
        volatility_annual: float | None = None,
        sl_atr_mult: float = 2.0,
    ) -> float:
        if equity <= 0 or price <= 0:
            return 0.0

        risk_amt = equity * (self.config.risk_per_trade_pct / 100.0)
        if self.config.sizing_mode == "atr_based" and atr_value and atr_value > 0:
            stop_dist = max(atr_value * sl_atr_mult, price * 0.002)
            qty = risk_amt / stop_dist
        elif self.config.sizing_mode == "volatility_based" and volatility_annual and volatility_annual > 0:
            daily_vol = volatility_annual / (365.0**0.5)
            stop_dist = max(price * daily_vol, price * 0.003)
            qty = risk_amt / stop_dist
        else:  # fixed_fractional
            notional = equity * min(self.config.max_asset_exposure_pct / 100.0, self.config.risk_per_trade_pct / 10.0)
            qty = notional / price

        max_asset_notional = equity * (self.config.max_asset_exposure_pct / 100.0) * self.config.max_leverage
        if qty * price > max_asset_notional:
            qty = max_asset_notional / price
        return max(0.0, float(qty))

    def validate_new_order(
        self,
        equity: float,
        peak_equity: float,
        daily_pnl: float,
        open_positions_count: int,
        asset_exposure_notional: float,
        order_notional: float,
        consecutive_losses: int,
    ) -> None:
        """Raise RiskViolationError if any risk control prevents opening a new position."""
        if self.config.emergency_halt:
            raise RiskViolationError("Emergency trading halt is active")

        if peak_equity > 0:
            dd_pct = (peak_equity - equity) / peak_equity * 100.0
            if dd_pct >= self.config.max_portfolio_drawdown_pct:
                raise RiskViolationError(f"Portfolio drawdown {dd_pct:.2f}% reached limit {self.config.max_portfolio_drawdown_pct:.2f}%")

        if equity > 0 and daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / equity * 100.0
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                raise RiskViolationError(f"Daily loss {daily_loss_pct:.2f}% reached limit {self.config.max_daily_loss_pct:.2f}%")

        if open_positions_count >= self.config.max_open_positions:
            raise RiskViolationError(f"Open positions {open_positions_count} at max limit {self.config.max_open_positions}")

        if consecutive_losses >= self.config.max_consecutive_losses:
            raise RiskViolationError(f"Consecutive loss protection triggered ({consecutive_losses} losses)")

        max_notional = equity * (self.config.max_asset_exposure_pct / 100.0) * self.config.max_leverage
        if asset_exposure_notional + order_notional > max_notional * 1.01:
            raise RiskViolationError("Order would exceed maximum asset exposure or leverage limit")
