"""Exchange adapter interfaces and canonical market data schemas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

SUPPORTED_TIMEFRAMES: tuple[str, ...] = tuple(TIMEFRAME_MS.keys())

DEFAULT_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


@dataclass(frozen=True, slots=True)
class CandleRow:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades_count: int
    taker_buy_volume: float
    taker_buy_quote_volume: float


class ExchangeAdapter(ABC):
    """Abstract configurable exchange adapter so additional exchanges can be plugged in."""

    exchange_name: str = "abstract"

    @abstractmethod
    def fetch_klines_page(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[CandleRow]:
        """Fetch a single page of OHLCV candles from the exchange."""

    @abstractmethod
    def fetch_latest_price(self, symbol: str, market_type: str = "spot") -> float:
        """Fetch current ticker price."""

    @abstractmethod
    def fetch_exchange_symbol_info(self, symbol: str, market_type: str = "spot") -> dict[str, Any]:
        """Fetch symbol tick size, step size, and trading constraints."""

    @abstractmethod
    def fetch_order_book_snapshot(self, symbol: str, market_type: str = "spot", limit: int = 20) -> dict[str, Any]:
        """Fetch live order book depth snapshot."""

    @abstractmethod
    def fetch_recent_agg_trades(
        self,
        symbol: str,
        market_type: str = "spot",
        start_time_ms: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch aggregate market trades."""
