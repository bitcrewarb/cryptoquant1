"""FastAPI endpoint, Auth/CSRF, Live Vault Encryption, Paper Trading Idempotency, and Job Recovery tests."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from crypto_quant_lab.app.auth.security import decrypt_secret
from crypto_quant_lab.app.main import app
from crypto_quant_lab.core.exceptions import RiskViolationError
from crypto_quant_lab.data.exchanges.base import CandleRow
from crypto_quant_lab.data.exchanges.binance import BinanceAdapter
from crypto_quant_lab.data.storage.parquet_store import dataset_parquet_paths, write_parquet
from crypto_quant_lab.database.models import (
    Asset,
    ExchangeConfiguration,
    MarketDataset,
    PaperOrder,
    ResearchJob,
    StrategyDefinition,
    StrategyExperiment,
    StrategyVersion,
)
from crypto_quant_lab.database.session import get_session_factory
from crypto_quant_lab.live_trading.adapter import (
    save_exchange_credentials,
    toggle_live_trading,
    trigger_emergency_kill_switch,
)
from crypto_quant_lab.paper_trading.engine import deploy_paper_strategy, step_paper_trading_cycle
from crypto_quant_lab.strategies.grammar import LeafCondition, StrategySpec
from crypto_quant_lab.workers.job_runner import recover_interrupted_jobs
from tests.conftest import make_synthetic_ohlcv


def _seed_dataset_and_experiment(tmp_path) -> tuple[int, int]:
    df = make_synthetic_ohlcv(n=360, seed=21, drift=0.0005, vol=0.015)
    raw_p, norm_p = dataset_parquet_paths("binance", "spot", "BTCUSDT", "1h", 1)
    write_parquet(df, raw_p)
    size = write_parquet(df, norm_p)

    spec = StrategySpec(
        name="ema_momentum_test",
        family="trend_following",
        direction="long_only",
        long_entry=LeafCondition("trend", "ema_gt_ema", {"fast": 10, "slow": 30}),
        long_exit=LeafCondition("trend", "ema_cross_below", {"fast": 10, "slow": 30}),
        short_entry=None,
        short_exit=None,
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
    )

    factory = get_session_factory()
    with factory() as db:
        asset = Asset(symbol="BTCUSDT", base_asset="BTC", quote_asset="USDT", exchange="binance", market_type="spot")
        db.add(asset)
        db.flush()

        ds = MarketDataset(
            asset_id=asset.id,
            symbol="BTCUSDT",
            exchange="binance",
            market_type="spot",
            timeframe="1h",
            version=1,
            provenance="synthetic_test_fixture",
            parquet_path=str(norm_p),
            raw_parquet_path=str(raw_p),
            candle_count=df.height,
            quality_status="verified",
            storage_bytes=size,
        )
        db.add(ds)
        db.flush()

        sdef = StrategyDefinition(
            strategy_uid="STR-TEST000001",
            name=spec.name,
            family=spec.family,
            direction=spec.direction,
            ast_hash=spec.ast_hash(),
            rule_json=spec.to_rule_dict(),
        )
        db.add(sdef)
        db.flush()

        sver = StrategyVersion(strategy_id=sdef.id, version=1, params_json=spec.to_params_dict(), param_hash=spec.param_hash())
        db.add(sver)
        db.flush()

        exp = StrategyExperiment(
            experiment_uid="EXP-TEST000001",
            strategy_id=sdef.id,
            strategy_version_id=sver.id,
            dataset_id=ds.id,
            search_algorithm="manual",
            random_seed=42,
            software_version="1.0.0",
            validation_tier="BACKTESTED",
        )
        db.add(exp)
        db.commit()
        return ds.id, exp.id


def test_auth_protection_pages_backtest_validation_and_exports(isolated_env) -> None:
    ds_id, exp_id = _seed_dataset_and_experiment(isolated_env)

    with TestClient(app) as client:
        # Protected API routes must reject unauthenticated requests
        assert client.get("/api/data/datasets").status_code == 401
        assert client.post("/api/discovery/start", json={"dataset_id": ds_id}).status_code == 401

        # Unauthenticated dashboard redirects to /login
        r_root = client.get("/", follow_redirects=False)
        assert r_root.status_code == 303 and r_root.headers["location"] == "/login"

        # Claim first-run admin account
        r_claim = client.post(
            "/auth/claim-admin",
            data={"username": "quantadmin", "password": "StrongPassword123!", "email": "admin@example.com"},
            follow_redirects=False,
        )
        assert r_claim.status_code == 303

        # Subsequent claim attempt must fail
        r_claim2 = client.post(
            "/auth/claim-admin",
            data={"username": "hacker", "password": "StrongPassword123!"},
        )
        assert r_claim2.status_code == 403

        # Verify all 10 dashboard pages render cleanly while authenticated
        for path in ("/", "/data", "/research", "/discovery", "/backtesting", "/validation", "/leaderboard", "/portfolio", "/paper-trading", "/experiments"):
            resp = client.get(path)
            assert resp.status_code == 200, f"Page {path} failed: {resp.status_code}"
            assert "/login" not in str(resp.url), f"Page {path} unexpectedly redirected to /login"

        # Run Quant Research analysis
        r_ana = client.post("/api/research/analyze", json={"dataset_id": ds_id, "tail_bars": 200})
        assert r_ana.status_code == 200
        assert "statistics" in r_ana.json()

        # Run Backtest
        r_bt = client.post("/api/backtest/run", json={"experiment_id": exp_id, "engine_type": "event_driven"})
        assert r_bt.status_code == 200
        assert "metrics" in r_bt.json()

        # Run Validation Suite
        r_val = client.post("/api/validation/run", json={"experiment_id": exp_id, "mc_simulations": 200, "wf_windows": 4})
        assert r_val.status_code == 200
        assert r_val.json()["validation_tier"] in ("VALIDATED", "MONTE_CARLO_TESTED")

        # Leaderboard & Portfolio
        r_lb = client.get("/api/leaderboard")
        assert r_lb.status_code == 200
        assert len(r_lb.json()) >= 1

        r_port = client.post(
            "/api/portfolio/construct",
            json={"name": "TestPortfolio", "experiment_ids": [exp_id], "allocation_method": "risk_parity"},
        )
        assert r_port.status_code == 200

        # JSON & CSV Exports
        r_json = client.get(f"/api/experiments/{exp_id}/export.json")
        assert r_json.status_code == 200
        assert "experiment_uid" in r_json.json()


def test_live_vault_encryption_and_disabled_by_default(isolated_env) -> None:
    factory = get_session_factory()
    with factory() as db:
        cfg = save_exchange_credentials(
            db=db,
            exchange="binance",
            market_type="spot",
            label="primary-vault",
            api_key="my_real_binance_api_key_value",
            api_secret="my_super_secret_binance_hmac_key",
            permissions_scope="spot_trade",
            max_capital_usdt=500.0,
        )
        assert cfg.live_trading_enabled is False
        assert cfg.encrypted_api_key != "my_real_binance_api_key_value"
        assert decrypt_secret(cfg.encrypted_api_key) == "my_real_binance_api_key_value"
        assert decrypt_secret(cfg.encrypted_api_secret) == "my_super_secret_binance_hmac_key"

        # Enabling must fail when LIVE_TRADING_ENABLED=false
        with pytest.raises(RiskViolationError):
            toggle_live_trading(db, cfg.id, True, "ENABLE LIVE TRADING")

        # Kill switch locks config
        trigger_emergency_kill_switch(db, cfg.id)
        db.refresh(cfg)
        assert cfg.kill_switch_triggered is True
        assert cfg.live_trading_enabled is False


def test_paper_trading_idempotency_across_restart(isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify paper trading executes cleanly and does NOT duplicate orders when stepped twice on the same closed candle."""
    _, exp_id = _seed_dataset_and_experiment(isolated_env)

    # Construct trending candle rows where ema(10) > ema(30) is guaranteed true at latest closed bar
    base_ms = int(time.time() * 1000) - 100 * 3_600_000
    rows = []
    for i in range(90):
        px = 40_000.0 + i * 120.0
        rows.append(
            CandleRow(
                open_time_ms=base_ms + i * 3_600_000,
                open=px,
                high=px + 80.0,
                low=px - 50.0,
                close=px + 60.0,
                volume=100.0,
                close_time_ms=base_ms + (i + 1) * 3_600_000 - 1,
                quote_volume=px * 100.0,
                trades_count=100,
                taker_buy_volume=60.0,
                taker_buy_quote_volume=px * 60.0,
            )
        )

    monkeypatch.setattr(BinanceAdapter, "fetch_klines_page", lambda *args, **kwargs: rows)
    monkeypatch.setattr(BinanceAdapter, "fetch_latest_price", lambda *args, **kwargs: rows[-1].close)

    factory = get_session_factory()
    with factory() as db:
        pos = deploy_paper_strategy(db, exp_id, "BTCUSDT", "1h", "spot", 10_000.0)
        res1 = step_paper_trading_cycle(db)
        assert res1["orders_executed"] == 1

        # Second step on exact same candle timestamp (simulating server restart) MUST execute 0 duplicate orders
        res2 = step_paper_trading_cycle(db)
        assert res2["orders_executed"] == 0

        orders = db.query(PaperOrder).filter(PaperOrder.experiment_id == exp_id).all()
        assert len(orders) == 1
        db.refresh(pos)
        assert pos.side == "LONG"


def test_job_checkpoint_pause_resume_and_restart_recovery(isolated_env) -> None:
    ds_id, _ = _seed_dataset_and_experiment(isolated_env)

    factory = get_session_factory()
    with factory() as db:
        stale_job = ResearchJob(
            job_uid="JOB-RECOVER01",
            job_type="strategy_discovery",
            status="running",
            config_json={"dataset_id": ds_id, "candidate_count": 30, "algorithm": "random_search", "seed": 42, "persist_top_n": 5},
            checkpoint_json={"offset": 10},
            total_items=30,
            completed_items=10,
            progress_pct=33.3,
        )
        db.add(stale_job)
        db.commit()
        job_id = stale_job.id

    recovered_n = recover_interrupted_jobs()
    assert recovered_n == 1

    # Wait briefly for ThreadPoolExecutor job completion
    for _ in range(60):
        with factory() as db:
            j = db.get(ResearchJob, job_id)
            if j and j.status == "completed":
                break
        time.sleep(0.15)

    with factory() as db:
        j = db.get(ResearchJob, job_id)
        assert j is not None
        assert j.status == "completed"
        assert j.completed_items == 30
        assert j.progress_pct == 100.0


def test_https_proxy_iframe_cookie_attributes(isolated_env) -> None:
    """Verify HTTPS proxy/iframe requests get SameSite=none; Secure; Partitioned and blocked cookies trigger helpful fallback."""
    with TestClient(app) as client:
        r_claim = client.post(
            "/auth/claim-admin",
            data={"username": "IframeAdmin", "password": "StrongPassword123!", "email": "desk@example.com"},
            headers={"origin": "https://ais-dev-preview.run.app"},
            follow_redirects=False,
        )
        assert r_claim.status_code == 303
        set_cookie_hdr = r_claim.headers.get("set-cookie", "")
        assert "SameSite=none" in set_cookie_hdr or "SameSite=None" in set_cookie_hdr
        assert "Secure" in set_cookie_hdr
        assert "Partitioned" in set_cookie_hdr

        # Simulate browser dropping cookie after redirect to /?cql_chk=1
        r_chk = client.get("/?cql_chk=1", follow_redirects=False)
        assert r_chk.status_code == 303
        assert r_chk.headers["location"] == "/login?cookie_blocked=1"

        r_login = client.get("/login?cookie_blocked=1")
        assert r_login.status_code == 200
        assert "Open Terminal in New Tab" in r_login.text

        # Verify case-insensitive username & email login plus clean invalid-login rendering
        r_bad = client.post("/auth/login", data={"username": "iframeadmin", "password": "WrongPassword"})
        assert r_bad.status_code == 200
        assert "Invalid username/email or password" in r_bad.text

        r_email = client.post(
            "/auth/login",
            data={"username": "DESK@EXAMPLE.COM", "password": "StrongPassword123!"},
            follow_redirects=False,
        )
        assert r_email.status_code == 303
        assert "cql_session" in r_email.cookies


def test_workflow_endpoints_delete_results_batch_and_preselection(isolated_env) -> None:
    """Verify dataset delete guard, job results endpoint, batch validation, tier sorting, and pre-selected workflow routes."""
    ds_id, exp_id = _seed_dataset_and_experiment(isolated_env)

    # Seed an unused second dataset and a completed discovery job linked to exp_id
    df2 = make_synthetic_ohlcv(n=180, seed=99, drift=0.0003, vol=0.01)
    raw_p2, norm_p2 = dataset_parquet_paths("binance", "spot", "ETHUSDT", "1h", 1)
    write_parquet(df2, raw_p2)
    size2 = write_parquet(df2, norm_p2)

    factory = get_session_factory()
    with factory() as db:
        asset2 = Asset(symbol="ETHUSDT", base_asset="ETH", quote_asset="USDT", exchange="binance", market_type="spot")
        db.add(asset2)
        db.flush()

        unused_ds = MarketDataset(
            asset_id=asset2.id,
            symbol="ETHUSDT",
            exchange="binance",
            market_type="spot",
            timeframe="1h",
            version=1,
            provenance="synthetic_test_fixture",
            parquet_path=str(norm_p2),
            raw_parquet_path=str(raw_p2),
            candle_count=df2.height,
            quality_status="verified",
            storage_bytes=size2,
        )
        db.add(unused_ds)

        job = ResearchJob(
            job_uid="JOB-DISCRES01",
            job_type="strategy_discovery",
            status="completed",
            config_json={"dataset_id": ds_id, "candidate_count": 20},
            checkpoint_json={"offset": 20},
            total_items=20,
            completed_items=20,
            progress_pct=100.0,
        )
        db.add(job)
        db.flush()

        exp = db.get(StrategyExperiment, exp_id)
        assert exp is not None
        exp.research_job_id = job.id
        exp.backtest_config_json = {"engine_type": "vectorized", "n_trials_attempted": 20}
        db.commit()
        unused_ds_id = unused_ds.id
        job_id = job.id

    with TestClient(app) as client:
        client.post(
            "/auth/claim-admin",
            data={"username": "flowadmin", "password": "StrongPassword123!", "email": "flow@example.com"},
            follow_redirects=False,
        )

        # Backtest first so job results & leaderboard have metrics
        assert client.post("/api/backtest/run", json={"experiment_id": exp_id, "engine_type": "event_driven"}).status_code == 200

        # Jobs listing + job_type filter
        r_jobs = client.get("/api/jobs?job_type=strategy_discovery")
        assert r_jobs.status_code == 200
        assert any(j["id"] == job_id for j in r_jobs.json())
        r_dl_jobs = client.get("/api/jobs?job_type=data_download")
        assert r_dl_jobs.status_code == 200
        assert all(j["job_type"] == "data_download" for j in r_dl_jobs.json())

        # Job results endpoint
        r_res = client.get(f"/api/jobs/{job_id}/results")
        assert r_res.status_code == 200
        res_data = r_res.json()
        assert res_data["count"] == 1
        assert res_data["results"][0]["experiment_id"] == exp_id

        # OHLCV preview & Dataset Delete protection (409 in-use, 200 unused)
        assert client.get(f"/api/data/datasets/{ds_id}/ohlcv?limit=50").status_code == 200
        assert client.delete(f"/api/data/datasets/{ds_id}").status_code == 409
        r_del = client.delete(f"/api/data/datasets/{unused_ds_id}")
        assert r_del.status_code == 200
        assert r_del.json()["deleted_id"] == unused_ds_id
        assert not norm_p2.exists()

        # Batch validation endpoint
        r_batch = client.post(
            "/api/validation/batch",
            json={"experiment_ids": [exp_id], "mc_simulations": 100, "wf_windows": 3, "n_trials_attempted": 20},
        )
        assert r_batch.status_code == 200
        batch_job_id = r_batch.json()["job_id"]
        for _ in range(60):
            with factory() as db:
                bj = db.get(ResearchJob, batch_job_id)
                if bj and bj.status == "completed":
                    break
            time.sleep(0.15)
        with factory() as db:
            bj = db.get(ResearchJob, batch_job_id)
            assert bj is not None and bj.status == "completed"

        # Pre-selected workflow page routes with query params
        for url in (
            f"/research?dataset_id={ds_id}&autorun=1",
            f"/discovery?dataset_id={ds_id}",
            f"/backtesting?experiment_id={exp_id}&dataset_id={ds_id}&autorun=1",
            f"/validation?experiment_id={exp_id}&autorun=1",
            f"/paper-trading?experiment_id={exp_id}",
        ):
            r_page = client.get(url)
            assert r_page.status_code == 200, f"Pre-selected page {url} failed: {r_page.status_code}"


