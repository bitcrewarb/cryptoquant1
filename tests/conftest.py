"""Shared pytest fixtures for CRYPTO QUANT LAB tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from crypto_quant_lab.app.config import get_settings
from crypto_quant_lab.database.session import init_db, reset_engine


def make_synthetic_ohlcv(
    n: int = 400,
    seed: int = 42,
    start_price: float = 50_000.0,
    drift: float = 0.0003,
    vol: float = 0.012,
    timeframe_ms: int = 3_600_000,
) -> pl.DataFrame:
    """Generate realistic deterministic synthetic OHLCV candles satisfying all OHLC invariants."""
    rng = np.random.default_rng(seed)
    start_ms = int(datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    times = np.arange(n, dtype=np.int64) * timeframe_ms + start_ms
    close_times = times + timeframe_ms - 1

    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    volumes = rng.uniform(50.0, 500.0, size=n)

    price = start_price
    for i in range(n):
        opens[i] = price
        # Mix regime shifts across segments so trend, range, and volatility signals fire naturally
        regime_drift = drift if (i // 80) % 2 == 0 else -drift * 0.6
        ret = regime_drift + rng.normal(0.0, vol)
        closes[i] = max(10.0, opens[i] * np.exp(ret))
        wick_up = abs(rng.normal(0.0, vol * 0.6)) * max(opens[i], closes[i])
        wick_dn = abs(rng.normal(0.0, vol * 0.6)) * min(opens[i], closes[i])
        highs[i] = max(opens[i], closes[i]) + wick_up
        lows[i] = max(1.0, min(opens[i], closes[i]) - wick_dn)
        price = closes[i]

    return pl.DataFrame(
        {
            "open_time_ms": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "close_time_ms": close_times,
            "quote_volume": volumes * closes,
            "trades_count": rng.integers(50, 1500, size=n),
            "taker_buy_volume": volumes * rng.uniform(0.35, 0.65, size=n),
            "taker_buy_quote_volume": volumes * closes * 0.5,
        }
    )


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide an isolated SQLite DB and data_store directory per test."""
    db_file = tmp_path / "test_cql.db"
    data_dir = tmp_path / "data_store"
    data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_STORE_DIR", str(data_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-pytest-only-32chars")
    monkeypatch.setenv("CREDENTIALS_ENCRYPTION_KEY", "test-fernet-seed-for-pytest-only")
    monkeypatch.setenv("ADMIN_PASSCODE", "")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")

    get_settings.cache_clear()
    reset_engine(f"sqlite:///{db_file}")
    init_db()
    yield tmp_path
    get_settings.cache_clear()
