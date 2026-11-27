"""Binance Spot and USDT-Margined Futures public market data adapter with automatic pagination and rate-limit safety."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from crypto_quant_lab.app.config import get_settings
from crypto_quant_lab.core.exceptions import ExchangeAPIError
from crypto_quant_lab.data.exchanges.base import TIMEFRAME_MS, CandleRow, ExchangeAdapter

logger = logging.getLogger("crypto_quant_lab.data.binance")


class BinanceAdapter(ExchangeAdapter):
    """Public REST adapter for Binance Spot (`api.binance.com`) and USDT-M Futures (`fapi.binance.com`)."""

    exchange_name = "binance"

    def __init__(
        self,
        spot_base_url: str | None = None,
        futures_base_url: str | None = None,
        timeout: float = 20.0,
        max_retries: int = 5,
    ) -> None:
        settings = get_settings()
        self.spot_base_url = (spot_base_url or settings.binance_spot_base_url).rstrip("/")
        self.futures_base_url = (futures_base_url or settings.binance_futures_base_url).rstrip("/")
        self.max_retries = max_retries
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "CryptoQuantLab/1.0"},
        )

    def close(self) -> None:
        self.client.close()

    def _base_url(self, market_type: str) -> str:
        return self.futures_base_url if market_type == "futures" else self.spot_base_url

    def _request(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{base_url}{path}"
        backoff = 0.6
        for attempt in range(self.max_retries):
            try:
                resp = self.client.get(url, params=params)
                # Respect Binance weight headers proactively
                used_weight = resp.headers.get("X-MBX-USED-WEIGHT-1M")
                if used_weight and used_weight.isdigit() and int(used_weight) > 1050:
                    time.sleep(1.5)

                if resp.status_code in (429, 418):
                    retry_after = float(resp.headers.get("Retry-After", backoff * 2))
                    logger.warning("Binance rate limit %s on %s, sleeping %.2fs", resp.status_code, path, retry_after)
                    time.sleep(min(retry_after, 30.0))
                    backoff *= 2
                    continue

                if resp.status_code >= 500:
                    logger.warning("Binance server error %s on %s (attempt %s)", resp.status_code, path, attempt + 1)
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                if resp.status_code >= 400:
                    raise ExchangeAPIError(
                        f"Binance HTTP {resp.status_code}: {resp.text[:300]}",
                        status_code=resp.status_code,
                    )

                return resp.json()
            except httpx.RequestError as exc:
                if attempt == self.max_retries - 1:
                    raise ExchangeAPIError(f"Network error contacting Binance ({url}): {exc}") from exc
                time.sleep(backoff)
                backoff *= 2
        raise ExchangeAPIError(f"Exhausted retries requesting {url}")

    def fetch_klines_page(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[CandleRow]:
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        path = "/fapi/v1/klines" if market_type == "futures" else "/api/v3/klines"
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": min(max(1, limit), 1000 if market_type == "spot" else 1500),
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        raw = self._request(self._base_url(market_type), path, params=params)
        rows: list[CandleRow] = []
        for item in raw:
            rows.append(
                CandleRow(
                    open_time_ms=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    close_time_ms=int(item[6]),
                    quote_volume=float(item[7]),
                    trades_count=int(item[8]),
                    taker_buy_volume=float(item[9]),
                    taker_buy_quote_volume=float(item[10]),
                )
            )
        return rows

    def download_klines_paginated(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        max_candles: int | None = None,
        progress_cb: Callable[[int, int | None], None] | None = None,
    ) -> list[CandleRow]:
        """Download historical OHLCV with automatic forward/backward pagination."""
        tf_ms = TIMEFRAME_MS[timeframe]
        now_ms = int(time.time() * 1000)
        resolved_end = end_time_ms if end_time_ms is not None else now_ms

        # If start_time_ms is omitted and max_candles is specified, paginate backward or calculate start
        if start_time_ms is None:
            if max_candles is not None:
                resolved_start = max(0, resolved_end - max_candles * tf_ms)
            else:
                # Probe earliest available candle on exchange using startTime=0
                first_page = self.fetch_klines_page(
                    symbol=symbol,
                    timeframe=timeframe,
                    market_type=market_type,
                    start_time_ms=0,
                    limit=1,
                )
                if not first_page:
                    return []
                resolved_start = first_page[0].open_time_ms
        else:
            resolved_start = start_time_ms

        all_rows: list[CandleRow] = []
        cursor = resolved_start
        page_limit = 1000
        seen_times: set[int] = set()

        while cursor <= resolved_end:
            batch = self.fetch_klines_page(
                symbol=symbol,
                timeframe=timeframe,
                market_type=market_type,
                start_time_ms=cursor,
                end_time_ms=resolved_end,
                limit=page_limit,
            )
            if not batch:
                break

            added = 0
            for row in batch:
                if row.open_time_ms not in seen_times and row.open_time_ms <= resolved_end:
                    seen_times.add(row.open_time_ms)
                    all_rows.append(row)
                    added += 1

            if progress_cb:
                progress_cb(len(all_rows), max_candles)

            if max_candles is not None and len(all_rows) >= max_candles:
                all_rows = all_rows[:max_candles]
                break

            last_open = batch[-1].open_time_ms
            next_cursor = last_open + tf_ms
            if next_cursor <= cursor or added == 0 or len(batch) < page_limit:
                # Check if there is a gap after batch; if len < page_limit we reached end_time_ms
                if len(batch) < page_limit:
                    break
                cursor = next_cursor
            else:
                cursor = next_cursor

            time.sleep(0.06)  # Gentle pacing within Binance 1200/min weight limit

        all_rows.sort(key=lambda r: r.open_time_ms)
        return all_rows

    def fetch_latest_price(self, symbol: str, market_type: str = "spot") -> float:
        path = "/fapi/v1/ticker/price" if market_type == "futures" else "/api/v3/ticker/price"
        data = self._request(self._base_url(market_type), path, params={"symbol": symbol.upper()})
        return float(data["price"])

    def fetch_exchange_symbol_info(self, symbol: str, market_type: str = "spot") -> dict[str, Any]:
        path = "/fapi/v1/exchangeInfo" if market_type == "futures" else "/api/v3/exchangeInfo"
        params = {} if market_type == "futures" else {"symbol": symbol.upper()}
        data = self._request(self._base_url(market_type), path, params=params)
        for sym_info in data.get("symbols", []):
            if sym_info.get("symbol") == symbol.upper():
                tick_size = 0.01
                step_size = 0.0001
                min_notional = 5.0
                for flt in sym_info.get("filters", []):
                    ftype = flt.get("filterType")
                    if ftype == "PRICE_FILTER":
                        tick_size = float(flt.get("tickSize", tick_size))
                    elif ftype == "LOT_SIZE":
                        step_size = float(flt.get("stepSize", step_size))
                    elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                        min_notional = float(flt.get("minNotional") or flt.get("notional") or min_notional)
                return {
                    "symbol": sym_info["symbol"],
                    "base_asset": sym_info["baseAsset"],
                    "quote_asset": sym_info["quoteAsset"],
                    "tick_size": tick_size,
                    "step_size": step_size,
                    "min_notional": min_notional,
                }
        raise ExchangeAPIError(f"Symbol {symbol} not found on Binance {market_type}")

    def fetch_funding_rates(
        self,
        symbol: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch historical USDT-M Futures funding rates (paginated up to 1000/call)."""
        all_items: list[dict[str, Any]] = []
        cursor = start_time_ms
        while True:
            params: dict[str, Any] = {"symbol": symbol.upper(), "limit": min(limit, 1000)}
            if cursor is not None:
                params["startTime"] = int(cursor)
            if end_time_ms is not None:
                params["endTime"] = int(end_time_ms)
            batch = self._request(self.futures_base_url, "/fapi/v1/fundingRate", params=params)
            if not batch:
                break
            all_items.extend(batch)
            if len(batch) < 1000:
                break
            cursor = int(batch[-1]["fundingTime"]) + 1
            if end_time_ms is not None and cursor > end_time_ms:
                break
            time.sleep(0.05)
        return [
            {
                "symbol": item["symbol"],
                "funding_time_ms": int(item["fundingTime"]),
                "funding_rate": float(item["fundingRate"]),
                "mark_price": float(item["markPrice"]) if item.get("markPrice") else None,
            }
            for item in all_items
        ]

    def fetch_open_interest_history(
        self,
        symbol: str,
        period: str = "1h",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch Binance Futures open interest history (Binance retains ~30 days on public REST)."""
        valid_periods = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
        resolved_period = period if period in valid_periods else "1h"
        data = self._request(
            self.futures_base_url,
            "/futures/data/openInterestHist",
            params={"symbol": symbol.upper(), "period": resolved_period, "limit": min(limit, 500)},
        )
        return [
            {
                "symbol": item["symbol"],
                "period": resolved_period,
                "timestamp_ms": int(item["timestamp"]),
                "open_interest": float(item["sumOpenInterest"]),
                "open_interest_value": float(item["sumOpenInterestValue"]),
            }
            for item in data
        ]

    def fetch_long_short_ratio(self, symbol: str, period: str = "1h", limit: int = 200) -> list[dict[str, Any]]:
        """Fetch global long/short account ratio (trailing 30d limit from Binance public REST)."""
        valid_periods = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
        resolved_period = period if period in valid_periods else "1h"
        data = self._request(
            self.futures_base_url,
            "/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol.upper(), "period": resolved_period, "limit": min(limit, 500)},
        )
        return [
            {
                "symbol": item["symbol"],
                "timestamp_ms": int(item["timestamp"]),
                "long_short_ratio": float(item["longShortRatio"]),
                "long_account": float(item["longAccount"]),
                "short_account": float(item["shortAccount"]),
            }
            for item in data
        ]

    def fetch_premium_index(self, symbol: str) -> dict[str, Any]:
        """Fetch current mark price, index price, and futures basis."""
        data = self._request(self.futures_base_url, "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()})
        mark = float(data["markPrice"])
        index = float(data["indexPrice"])
        basis_bps = ((mark - index) / index * 10_000.0) if index > 0 else 0.0
        return {
            "symbol": data["symbol"],
            "mark_price": mark,
            "index_price": index,
            "basis_bps": basis_bps,
            "last_funding_rate": float(data.get("lastFundingRate", 0.0)),
            "next_funding_time_ms": int(data.get("nextFundingTime", 0)),
        }

    def fetch_order_book_snapshot(self, symbol: str, market_type: str = "spot", limit: int = 20) -> dict[str, Any]:
        """Fetch live order book depth snapshot (cannot be fetched historically via Binance REST)."""
        path = "/fapi/v1/depth" if market_type == "futures" else "/api/v3/depth"
        data = self._request(self._base_url(market_type), path, params={"symbol": symbol.upper(), "limit": limit})
        bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
        asks = [[float(p), float(q)] for p, q in data.get("asks", [])]
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0.0
        spread_bps = ((best_ask - best_bid) / mid * 10_000.0) if mid > 0 else 0.0
        bid_depth_10 = sum(q for _, q in bids[:10])
        ask_depth_10 = sum(q for _, q in asks[:10])
        denom = bid_depth_10 + ask_depth_10
        imbalance = ((bid_depth_10 - ask_depth_10) / denom) if denom > 0 else 0.0
        return {
            "symbol": symbol.upper(),
            "market_type": market_type,
            "collection_mode": "live_snapshot",
            "timestamp_ms": int(time.time() * 1000),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": spread_bps,
            "bid_depth_10": bid_depth_10,
            "ask_depth_10": ask_depth_10,
            "imbalance": imbalance,
            "bids": bids[:10],
            "asks": asks[:10],
        }

    def fetch_recent_agg_trades(
        self,
        symbol: str,
        market_type: str = "spot",
        start_time_ms: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        path = "/fapi/v1/aggTrades" if market_type == "futures" else "/api/v3/aggTrades"
        params: dict[str, Any] = {"symbol": symbol.upper(), "limit": min(limit, 1000)}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        data = self._request(self._base_url(market_type), path, params=params)
        return [
            {
                "agg_trade_id": int(t["a"]),
                "price": float(t["p"]),
                "quantity": float(t["q"]),
                "timestamp_ms": int(t["T"]),
                "is_buyer_maker": bool(t["m"]),
            }
            for t in data
        ]
