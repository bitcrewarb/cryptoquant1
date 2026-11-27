"""Normalized SQLAlchemy 2.0 ORM schema for CRYPTO QUANT LAB (25 core domain tables)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from crypto_quant_lab.database.session import Base

PKBigInt = BigInteger().with_variant(Integer, "sqlite")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# 1. Users
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="researcher", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# 2. Exchange Configurations (encrypted credentials at rest; live trading disabled by default)
class ExchangeConfiguration(Base):
    __tablename__ = "exchange_configurations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(32), default="spot", nullable=False)  # spot | futures
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    encrypted_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_api_secret: Mapped[str | None] = mapped_column(Text, nullable=True)
    permissions_scope: Mapped[str] = mapped_column(String(32), default="public_only", nullable=False)
    live_trading_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    max_capital_usdt: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    max_exposure_pct: Mapped[float] = mapped_column(Float, default=10.0, nullable=False)
    kill_switch_triggered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("exchange", "market_type", "label", name="uq_exchange_cfg_label"),)


# 3. Assets
class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    base_asset: Mapped[str] = mapped_column(String(16), nullable=False)
    quote_asset: Mapped[str] = mapped_column(String(16), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), default="spot", nullable=False)
    tick_size: Mapped[float] = mapped_column(Float, default=0.01, nullable=False)
    step_size: Mapped[float] = mapped_column(Float, default=0.0001, nullable=False)
    min_notional: Mapped[float] = mapped_column(Float, default=5.0, nullable=False)
    maker_fee_bps: Mapped[float] = mapped_column(Float, default=10.0, nullable=False)
    taker_fee_bps: Mapped[float] = mapped_column(Float, default=10.0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    datasets: Mapped[list["MarketDataset"]] = relationship(back_populates="asset", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("symbol", "exchange", "market_type", name="uq_asset_symbol_ex_mkt"),)


# 4. Market Datasets (Parquet-backed dataset registry with versioning & provenance)
class MarketDataset(Base):
    __tablename__ = "market_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), default="spot", nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    provenance: Mapped[str] = mapped_column(String(64), default="binance_public_api", nullable=False)
    parquet_path: Mapped[str] = mapped_column(String(512), nullable=False)
    raw_parquet_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    candle_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    missing_candles: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duplicate_candles: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    outlier_candles: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    quality_report_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    storage_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    asset: Mapped["Asset"] = relationship(back_populates="datasets")

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "market_type", "timeframe", "version", name="uq_dataset_ver"),
        Index("ix_dataset_lookup", "symbol", "market_type", "timeframe"),
    )


# 5. Candles (Indexed relational OHLCV table for queryable windows / Timescale-ready)
class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("market_datasets.id", ondelete="CASCADE"), nullable=False)
    open_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    close_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    quote_volume: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    trades_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    taker_buy_volume: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    taker_buy_quote_volume: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    __table_args__ = (
        UniqueConstraint("dataset_id", "open_time_ms", name="uq_candle_ds_time"),
        Index("ix_candles_ds_time", "dataset_id", "open_time_ms"),
    )


# 6. Funding Rates
class FundingRate(Base):
    __tablename__ = "funding_rates"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", nullable=False)
    funding_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    funding_rate: Mapped[float] = mapped_column(Float, nullable=False)
    mark_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    index_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="binance_fapi", nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "funding_time_ms", name="uq_funding_sym_time"),
        Index("ix_funding_sym_time", "symbol", "funding_time_ms"),
    )


# 7. Open Interest
class OpenInterest(Base):
    __tablename__ = "open_interest"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", nullable=False)
    period: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    open_interest: Mapped[float] = mapped_column(Float, nullable=False)
    open_interest_value: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    long_short_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="binance_fapi", nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "period", "timestamp_ms", name="uq_oi_sym_period_time"),
        Index("ix_oi_sym_time", "symbol", "timestamp_ms"),
    )


# 8. Market Trades
class MarketTrade(Base):
    __tablename__ = "market_trades"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", nullable=False)
    market_type: Mapped[str] = mapped_column(String(16), default="spot", nullable=False)
    agg_trade_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    is_buyer_maker: Mapped[bool] = mapped_column(Boolean, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "market_type", "agg_trade_id", name="uq_mkt_trade_id"),
        Index("ix_mkt_trades_sym_time", "symbol", "timestamp_ms"),
    )


# 9. Order Book Snapshots (clearly tagged live_collected vs historical)
class OrderBookSnapshot(Base):
    __tablename__ = "order_book_snapshots"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", nullable=False)
    market_type: Mapped[str] = mapped_column(String(16), default="spot", nullable=False)
    collection_mode: Mapped[str] = mapped_column(String(32), default="live_snapshot", nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    best_bid: Mapped[float] = mapped_column(Float, nullable=False)
    best_ask: Mapped[float] = mapped_column(Float, nullable=False)
    spread_bps: Mapped[float] = mapped_column(Float, nullable=False)
    bid_depth_10: Mapped[float] = mapped_column(Float, nullable=False)
    ask_depth_10: Mapped[float] = mapped_column(Float, nullable=False)
    imbalance: Mapped[float] = mapped_column(Float, nullable=False)
    levels_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    __table_args__ = (Index("ix_ob_sym_time", "symbol", "timestamp_ms"),)


# 10. Strategy Definitions
class StrategyDefinition(Base):
    __tablename__ = "strategy_definitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    family: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(16), default="long_short", nullable=False)
    ast_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    rule_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    versions: Mapped[list["StrategyVersion"]] = relationship(back_populates="definition", cascade="all, delete-orphan")


# 11. Strategy Versions
class StrategyVersion(Base):
    __tablename__ = "strategy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategy_definitions.id", ondelete="CASCADE"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    param_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    definition: Mapped["StrategyDefinition"] = relationship(back_populates="versions")

    __table_args__ = (UniqueConstraint("strategy_id", "version", name="uq_strat_version"),)


# 12. Research Jobs (persistent checkpoints for pause/resume/recovery)
class ResearchJob(Base):
    __tablename__ = "research_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False, index=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    checkpoint_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    total_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# 13. Strategy Experiments (full reproducibility record)
class StrategyExperiment(Base):
    __tablename__ = "strategy_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    research_job_id: Mapped[int | None] = mapped_column(ForeignKey("research_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategy_definitions.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_version_id: Mapped[int] = mapped_column(ForeignKey("strategy_versions.id", ondelete="CASCADE"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("market_datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    search_algorithm: Mapped[str] = mapped_column(String(48), default="random_search", nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, default=42, nullable=False)
    software_version: Mapped[str] = mapped_column(String(32), default="1.0.0", nullable=False)
    backtest_config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    validation_config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    validation_tier: Mapped[str] = mapped_column(String(32), default="BACKTESTED", nullable=False, index=True)
    composite_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 14. Backtest Results
class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("strategy_experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    engine_type: Mapped[str] = mapped_column(String(32), default="vectorized", nullable=False)  # vectorized | event_driven
    split_segment: Mapped[str] = mapped_column(String(32), default="train", nullable=False)  # train | validation | test | full
    net_profit: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    gross_profit: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    gross_loss: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    expectancy: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sharpe_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, index=True)
    sortino_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    calmar_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    recovery_factor: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    ulcer_index: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    return_on_capital_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    annualized_return_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    exposure_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    equity_curve_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    regime_breakdown_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 15. Trade Records
class TradeRecord(Base):
    __tablename__ = "trade_records"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    backtest_id: Mapped[int] = mapped_column(ForeignKey("backtest_results.id", ondelete="CASCADE"), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # LONG | SHORT
    entry_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    exit_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    gross_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    fee_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    slippage_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    funding_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    net_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(32), nullable=False)
    regime_at_entry: Mapped[str] = mapped_column(String(48), default="unknown", nullable=False)


# 16. Validation Results
class ValidationResult(Base):
    __tablename__ = "validation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("strategy_experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    oos_sharpe: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    oos_return_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    oos_max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    pbo_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    deflated_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    fdr_q_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    param_stability_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    stress_survival_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    passed_gates: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    rejection_reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 17. Monte Carlo Results
class MonteCarloResult(Base):
    __tablename__ = "monte_carlo_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("strategy_experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(48), default="trade_permutation", nullable=False)
    simulations: Mapped[int] = mapped_column(Integer, default=1000, nullable=False)
    median_return_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    p05_return_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    p95_return_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    median_max_dd_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    p95_max_dd_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    ruin_probability: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 18. Walk-Forward Results
class WalkForwardResult(Base):
    __tablename__ = "walk_forward_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("strategy_experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(32), default="rolling", nullable=False)  # rolling | expanding | anchored
    num_windows: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    consistency_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_oos_sharpe: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    wf_efficiency_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    windows_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 19. Market Regimes
class MarketRegime(Base):
    __tablename__ = "market_regimes"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("market_datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    open_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    primary_regime: Mapped[str] = mapped_column(String(48), nullable=False)
    trend_regime: Mapped[str] = mapped_column(String(48), nullable=False)
    volatility_regime: Mapped[str] = mapped_column(String(48), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)

    __table_args__ = (
        UniqueConstraint("dataset_id", "open_time_ms", name="uq_regime_ds_time"),
        Index("ix_regime_ds_time", "dataset_id", "open_time_ms"),
    )


# 20. Portfolio Definitions
class PortfolioDefinition(Base):
    __tablename__ = "portfolio_definitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    allocation_method: Mapped[str] = mapped_column(String(48), default="risk_parity", nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, default=100000.0, nullable=False)
    risk_config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    correlation_matrix_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    allocations: Mapped[list["PortfolioAllocation"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")


# 21. Portfolio Allocations
class PortfolioAllocation(Base):
    __tablename__ = "portfolio_allocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolio_definitions.id", ondelete="CASCADE"), nullable=False, index=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("strategy_experiments.id", ondelete="CASCADE"), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    risk_contribution: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    portfolio: Mapped["PortfolioDefinition"] = relationship(back_populates="allocations")


# 22. Paper Orders
class PaperOrder(Base):
    __tablename__ = "paper_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("strategy_experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY | SELL
    order_type: Mapped[str] = mapped_column(String(16), default="MARKET", nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    slippage_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="FILLED", nullable=False)
    candle_open_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 23. Paper Positions
class PaperPosition(Base):
    __tablename__ = "paper_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("strategy_experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), default="1h", nullable=False)
    market_type: Mapped[str] = mapped_column(String(16), default="spot", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), default="FLAT", nullable=False)  # FLAT | LONG | SHORT
    quantity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mark_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    allocated_capital: Mapped[float] = mapped_column(Float, default=10000.0, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    peak_equity: Mapped[float] = mapped_column(Float, default=10000.0, nullable=False)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_processed_candle_ms: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("experiment_id", "symbol", name="uq_paper_pos_exp_sym"),)


# 24. Paper Trades (Closed virtual trade ledger)
class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("strategy_experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    exit_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    gross_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    fees_and_slippage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    net_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 25. System Logs
class SystemLog(Base):
    __tablename__ = "system_logs"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    module: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


# 26. Market Events (Structured detection events shared with charts and signal feed)
class MarketEvent(Base):
    __tablename__ = "market_events"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    exchange: Mapped[str] = mapped_column(String(32), default="binance", nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), default="1h", nullable=False)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(16), default="neutral", nullable=False)
    time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    bar_index: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    level: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 27. Zone Records (Persistent zone lifecycle tracking)
class ZoneRecord(Base):
    __tablename__ = "zone_records"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    zone_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), default="1h", nullable=False)
    zone_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # ob | fvg | sr | breaker | bpr
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    top: Mapped[float] = mapped_column(Float, nullable=False)
    bottom: Mapped[float] = mapped_column(Float, nullable=False)
    created_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_bar: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="fresh", nullable=False, index=True)  # fresh | tested | mitigated | invalidated | expired
    mitigated_time_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    expiry_bar: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strength: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    touches: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    meta_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# 28. Signal Records (Idempotent live/paper signal lifecycle)
class SignalRecord(Base):
    __tablename__ = "signal_records"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    signal_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategy_definitions.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), default="1h", nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)  # LONG | SHORT
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False, index=True)  # pending | triggered | tp1 | tp2 | sl | expired | invalidated
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False)
    created_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    triggered_time_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    closed_time_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    pnl_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    meta_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    updates: Mapped[list["SignalUpdate"]] = relationship(back_populates="signal", cascade="all, delete-orphan")
    deliveries: Mapped[list["AlertDelivery"]] = relationship(back_populates="signal", cascade="all, delete-orphan")


# 29. Signal Updates
class SignalUpdate(Base):
    __tablename__ = "signal_updates"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("signal_records.id", ondelete="CASCADE"), nullable=False, index=True)
    previous_status: Mapped[str] = mapped_column(String(32), nullable=False)
    new_status: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    signal: Mapped["SignalRecord"] = relationship(back_populates="updates")


# 30. Alert Deliveries
class AlertDelivery(Base):
    __tablename__ = "alert_deliveries"

    id: Mapped[int] = mapped_column(PKBigInt, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("signal_records.id", ondelete="CASCADE"), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)  # telegram | webhook | email
    recipient: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="sent", nullable=False)  # sent | failed
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    delivered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    signal: Mapped["SignalRecord"] = relationship(back_populates="deliveries")
