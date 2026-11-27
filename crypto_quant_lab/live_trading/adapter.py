"""Optional Live Trading Architecture (DISABLED BY DEFAULT, encrypted credentials at rest)."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_quant_lab.app.auth.security import decrypt_secret, encrypt_secret
from crypto_quant_lab.app.config import get_settings
from crypto_quant_lab.core.exceptions import RiskViolationError
from crypto_quant_lab.core.logging import log_system_event
from crypto_quant_lab.database.models import ExchangeConfiguration


def save_exchange_credentials(
    db: Session,
    exchange: str,
    market_type: str,
    label: str,
    api_key: str,
    api_secret: str,
    permissions_scope: str = "read_only",
    max_capital_usdt: float = 0.0,
    max_exposure_pct: float = 10.0,
) -> ExchangeConfiguration:
    """Store encrypted exchange API credentials; live trading remains disabled until explicit confirmation."""
    if permissions_scope == "withdraw":
        raise ValueError("Withdrawal permissions are strictly prohibited")

    cfg = db.scalar(
        select(ExchangeConfiguration).where(
            ExchangeConfiguration.exchange == exchange,
            ExchangeConfiguration.market_type == market_type,
            ExchangeConfiguration.label == label,
        )
    )
    if cfg is None:
        cfg = ExchangeConfiguration(
            exchange=exchange,
            market_type=market_type,
            label=label,
        )
        db.add(cfg)

    cfg.encrypted_api_key = encrypt_secret(api_key.strip())
    cfg.encrypted_api_secret = encrypt_secret(api_secret.strip())
    cfg.permissions_scope = permissions_scope
    cfg.max_capital_usdt = max_capital_usdt
    cfg.max_exposure_pct = max_exposure_pct
    cfg.live_trading_enabled = False  # Always disabled on save until explicit confirmation
    db.commit()
    db.refresh(cfg)
    log_system_event(db, "INFO", "live_trading.vault", f"Saved encrypted API configuration '{label}' (live=False)")
    return cfg


def toggle_live_trading(
    db: Session,
    config_id: int,
    enable: bool,
    explicit_confirmation_phrase: str = "",
) -> ExchangeConfiguration:
    cfg = db.get(ExchangeConfiguration, config_id)
    if cfg is None:
        raise ValueError("Exchange configuration not found")

    if enable:
        if not get_settings().live_trading_enabled:
            raise RiskViolationError("Global LIVE_TRADING_ENABLED environment flag is false; live execution locked")
        if explicit_confirmation_phrase.strip() != "ENABLE LIVE TRADING":
            raise RiskViolationError("Explicit confirmation phrase 'ENABLE LIVE TRADING' required")
        if cfg.kill_switch_triggered:
            raise RiskViolationError("Emergency kill switch is engaged; reset kill switch first")
        if cfg.max_capital_usdt <= 0:
            raise RiskViolationError("Set a positive maximum capital limit before enabling live trading")
        cfg.live_trading_enabled = True
        log_system_event(db, "WARNING", "live_trading", f"Live trading ENABLED for config {cfg.label}")
    else:
        cfg.live_trading_enabled = False
        log_system_event(db, "INFO", "live_trading", f"Live trading DISABLED for config {cfg.label}")

    db.commit()
    db.refresh(cfg)
    return cfg


def trigger_emergency_kill_switch(db: Session, config_id: int | None = None) -> None:
    if config_id is not None:
        cfg = db.get(ExchangeConfiguration, config_id)
        if cfg:
            cfg.kill_switch_triggered = True
            cfg.live_trading_enabled = False
    else:
        for cfg in db.scalars(select(ExchangeConfiguration)).all():
            cfg.kill_switch_triggered = True
            cfg.live_trading_enabled = False
    db.commit()
    log_system_event(db, "CRITICAL", "live_trading.kill_switch", "Emergency kill switch triggered; all live configs halted")


def check_exchange_connectivity(db: Session, config_id: int) -> dict[str, Any]:
    cfg = db.get(ExchangeConfiguration, config_id)
    if cfg is None:
        raise ValueError("Configuration not found")
    settings = get_settings()
    base = settings.binance_futures_base_url if cfg.market_type == "futures" else settings.binance_spot_base_url
    ping_path = "/fapi/v1/ping" if cfg.market_type == "futures" else "/api/v3/ping"
    t0 = time.perf_counter()
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{base}{ping_path}")
        latency_ms = (time.perf_counter() - t0) * 1000.0

    # Never expose raw keys; verify decryption works internally if keys present
    has_key = bool(decrypt_secret(cfg.encrypted_api_key))
    return {
        "reachable": resp.status_code == 200,
        "latency_ms": round(latency_ms, 1),
        "credentials_decryptable": has_key,
        "live_trading_enabled": cfg.live_trading_enabled,
        "kill_switch_triggered": cfg.kill_switch_triggered,
    }


def sign_binance_params(params: dict[str, Any], api_secret: str) -> str:
    query = urlencode(params)
    return hmac.new(api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
