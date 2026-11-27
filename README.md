# CRYPTO QUANT LAB

Institutional-grade quantitative cryptocurrency strategy research, discovery, backtesting, statistical validation, portfolio construction, and paper trading platform.

Core research and strategy discovery run deterministically using mathematical, statistical, technical, Price Action, Smart Money Concepts (SMC), and ICT models—with zero dependency on external AI APIs or paid LLM services.

---

## 1. Architecture Overview

- **Backend & Web**: Python 3.11+/3.12+, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic, Jinja2, HTMX, Plotly.js.
- **Data Engine**: Binance Spot (`api.binance.com`) & USDT-M Futures (`fapi.binance.com`) public REST adapters with automatic historical pagination, rate-limit backoff, derivatives ingestion (funding rates, open interest, long/short ratios, mark/index basis, live order book snapshots), quality audits, and versioned Parquet storage via Polars/PyArrow.
- **Quantitative Library**: Causal implementations of Trend (11), Momentum (9), Volatility (9), Volume (8), Statistical/Econometric (ADF, KPSS, Engle-Granger Cointegration, Kalman Filter, Hurst Exponent, Shannon Entropy, GARCH(1,1) MLE), Price Action, SMC (BOS/CHoCH/MSS/OB/Breaker/FVG/IFVG/Liquidity Sweeps), ICT (Kill Zones, OTE, Judas Swing, Power of Three, Consequent Encroachment), and 10-regime Market Regime Detection.
- **Strategy Discovery Engine**: Formal composable AST grammar with canonical deduplication & contradiction rejection, streaming 20,000+ unique valid candidate strategies across 14 families using Grid Search, Random Search, Genetic Algorithms, Evolutionary Search, or Bayesian Optuna TPE.
- **Dual Backtesting Engines**: High-throughput vectorized screening engine + detailed bar-by-bar event-driven execution simulator (Market/Limit/Stop orders, partial fills, trailing stops, 8h futures funding, conservative OHLC ambiguous SL/TP resolution) with 26 performance metrics.
- **Statistical Validation**: Chronological 60/20/20 Train/Validation/Untouched-Test split, Rolling/Expanding/Anchored Walk-Forward, 1,000+ Monte Carlo simulations (permutation, bootstrap, block bootstrap), CSCV Probability of Backtest Overfitting (PBO), Deflated Sharpe Ratio (DSR), Benjamini-Hochberg FDR, Parameter Stability, Stress Testing, and Weighted Ranking with hard rejection gates.
- **Portfolio & Risk**: Return/Drawdown correlation matrices, Ward hierarchical clustering, 6 allocation optimizers (Risk Parity, Minimum Variance, Volatility-Weighted, Sharpe-Weighted, Max-DD Constrained, Equal Weight), and execution-time RiskManager circuit breakers.
- **Paper & Live Architecture**: Real-time Binance public-feed paper trading with idempotent restart recovery, plus an encrypted live API vault (disabled by default; requires explicit confirmation and kill-switch safety).

---

## 2. Clean Installation & Local Development

### Prerequisites
- Python 3.11+ (3.12 recommended)
- Git

### Exact Local Commands
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install locked dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set SECRET_KEY and CREDENTIALS_ENCRYPTION_KEY

# 4. Run database migrations
alembic upgrade head

# 5. Start the web terminal directly on port 3000 (or `npm run dev` for tsx wrapper)
uvicorn crypto_quant_lab.app.main:app --host 0.0.0.0 --port 3000
```

Open `http://localhost:3000`. On first visit (if `ADMIN_PASSCODE` is unset), complete the inline **Initialize Administrator Account** form. Authentication, session cookies, CSRF origin validation, and Fernet credential encryption live in `crypto_quant_lab/app/auth/security.py`.

---

## 3. Linux VPS Production Deployment (Docker Compose + TimescaleDB + Redis + Celery + Nginx)

```bash
# 1. Clone and configure environment
cp .env.example .env
# Generate strong keys:
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('CREDENTIALS_ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"

# 2. Build and launch stack
docker compose up -d --build

# 3. Verify health (Alembic migrations run automatically on web container startup)
curl -s http://localhost/health
```

---

## 4. Operational Guides

### Historical Data Download & API Limitations
- Use **Market Data Center** (`/data`) to download Spot or USDT-M Futures OHLCV for `BTCUSDT`, `ETHUSDT`, `SOLUSDT` or register additional Binance pairs dynamically.
- **Known Exchange API Limitations (Documented Honestly)**:
  - Binance public REST `/futures/data/openInterestHist` and `/futures/data/globalLongShortAccountRatio` retain only the trailing ~30 days of data.
  - Binance public REST does not serve historical order book depth snapshots; order book snapshots collected by this platform are tagged `collection_mode="live_snapshot"` and stored separately from historical OHLCV datasets.

### Strategy Generation & Statistical Validation Methodology
- Strategies are generated lazily in **Strategy Discovery** (`/discovery`) on the chronological **training split (60%)** only.
- Candidate uniqueness is enforced via canonicalized AST SHA-256 hashing (`A AND B == B AND A`).
- **Validation Center** (`/validation`) evaluates strategies on the **validation split (20%)** using Event-Driven simulation, Rolling/Expanding/Anchored Walk-Forward, 1,000+ Monte Carlo runs, CSCV PBO, Deflated Sharpe Ratio (accounting for attempted trial count $N$), neighboring parameter stability, and 5 stress-test scenarios. The **final test split (20%)** remains untouched until explicit final verification.

### Paper Trading & Live Trading Safety
- **Paper Trading Terminal** (`/paper-trading`) streams closed candles from Binance public APIs and executes virtual orders idempotently (`idempotency_key` per experiment/symbol/candle/side) so server restarts never duplicate trades.
- Live trading is **disabled by default**. To configure optional live API credentials, set `CREDENTIALS_ENCRYPTION_KEY` in `.env`, store keys via the encrypted vault (never with withdrawal permissions), and keep `LIVE_TRADING_ENABLED=false` unless explicitly authorized.

---

## 5. Automated Testing

```bash
# Run full unit, property-based (Hypothesis), look-ahead bias, and integration test suite
pytest -v
```

### Troubleshooting
- **Database locked on SQLite**: WAL mode and a 5s busy timeout are enabled automatically; for concurrent multi-worker research workloads, switch `DATABASE_URL` to PostgreSQL/TimescaleDB via Docker Compose.
- **Binance HTTP 429 / Rate Limits**: `BinanceAdapter` inspects `X-MBX-USED-WEIGHT-1M` and performs exponential backoff automatically. Reduce concurrent download jobs if hitting IP weight limits.

---

## 6. Measured Performance Benchmarks

Measured locally in this environment (`.venv/bin/python`, single-process CPU, synthetic 1,000-bar 1h OHLCV series with pre-warmed `IndicatorCache`):

- **20,000 Unique Strategy Generation (`stream_candidate_strategies`, canonical AST deduplication & contradiction filtering)**: `16.42s` wall-clock time, `14.9 MB` peak traced Python RAM (`tracemalloc`).
- **Vectorized Screening Engine (`evaluate_strategy`, `engine_type="vectorized"`, 2,000 strategies over 1,000 bars)**: `6.45s` (`~310 strategies/sec`).

