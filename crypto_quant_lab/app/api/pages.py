"""Server-rendered Jinja2 page routes for all 10 CRYPTO QUANT LAB terminal modules."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from crypto_quant_lab.app.config import get_settings
from crypto_quant_lab.app.dependencies import get_optional_user
from crypto_quant_lab.data.exchanges.base import SUPPORTED_TIMEFRAMES
from crypto_quant_lab.database.models import (
    Asset,
    BacktestResult,
    ExchangeConfiguration,
    MarketDataset,
    MarketRegime,
    MonteCarloResult,
    PaperOrder,
    PaperPosition,
    PaperTrade,
    PortfolioDefinition,
    ResearchJob,
    StrategyDefinition,
    StrategyExperiment,
    SystemLog,
    User,
    ValidationResult,
    WalkForwardResult,
)
from crypto_quant_lab.database.session import get_db
from crypto_quant_lab.strategies.grammar import STRATEGY_FAMILIES

router = APIRouter()
templates = Jinja2Templates(directory="crypto_quant_lab/app/templates")


def _auth_redirect() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


def _base_ctx(request: Request, user: User, active_page: str, title: str) -> dict[str, object]:
    return {
        "request": request,
        "user": user,
        "active_page": active_page,
        "page_title": title,
    }


@router.get("/", response_class=HTMLResponse)
def overview_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        if request.query_params.get("cql_chk") == "1":
            return RedirectResponse(url="/login?cookie_blocked=1", status_code=status.HTTP_303_SEE_OTHER)
        return _auth_redirect()

    total_generated = db.scalar(select(func.count(StrategyDefinition.id))) or 0
    total_tested = db.scalar(select(func.count(StrategyExperiment.id))) or 0
    passed_val = db.scalar(select(func.count(ValidationResult.id)).where(ValidationResult.passed_gates.is_(True))) or 0
    failed_val = db.scalar(select(func.count(ValidationResult.id)).where(ValidationResult.passed_gates.is_(False))) or 0
    active_jobs = db.scalar(select(func.count(ResearchJob.id)).where(ResearchJob.status.in_(["queued", "running"]))) or 0

    paper_positions = list(db.scalars(select(PaperPosition).where(PaperPosition.is_active.is_(True))).all())
    active_paper = len(paper_positions)
    paper_equity = sum(p.allocated_capital + p.realized_pnl + p.unrealized_pnl for p in paper_positions)
    paper_peak = sum(p.peak_equity for p in paper_positions)
    current_dd = ((paper_peak - paper_equity) / paper_peak * 100.0) if paper_peak > 0 else 0.0

    latest_jobs = list(db.scalars(select(ResearchJob).order_by(desc(ResearchJob.created_at)).limit(8)).all())

    best_exps = list(
        db.scalars(
            select(StrategyExperiment)
            .where(StrategyExperiment.validation_tier == "VALIDATED")
            .order_by(desc(StrategyExperiment.composite_score))
            .limit(8)
        ).all()
    )
    if not best_exps:
        best_exps = list(db.scalars(select(StrategyExperiment).order_by(desc(StrategyExperiment.composite_score)).limit(8)).all())

    best_rows = []
    for e in best_exps:
        sdef = db.get(StrategyDefinition, e.strategy_id)
        ds = db.get(MarketDataset, e.dataset_id)
        bt = db.scalar(select(BacktestResult).where(BacktestResult.experiment_id == e.id).order_by(desc(BacktestResult.id)))
        if sdef and ds and bt:
            best_rows.append({"exp": e, "strat": sdef, "ds": ds, "bt": bt})

    dataset_count = db.scalar(select(func.count(MarketDataset.id))) or 0
    regime_count = db.scalar(select(func.count(func.distinct(MarketRegime.dataset_id)))) or 0
    portfolio_count = db.scalar(select(func.count(PortfolioDefinition.id))) or 0

    ctx = _base_ctx(request, user, "overview", "Overview Dashboard")
    ctx.update(
        {
            "stats": {
                "total_generated": total_generated,
                "total_tested": total_tested,
                "passed_validation": passed_val,
                "failed_validation": failed_val,
                "active_jobs": active_jobs,
                "active_paper": active_paper,
                "portfolio_equity": round(paper_equity, 2),
                "current_drawdown": round(max(0.0, current_dd), 2),
                "dataset_count": dataset_count,
                "regime_count": regime_count,
                "portfolio_count": portfolio_count,
            },
            "latest_jobs": latest_jobs,
            "best_rows": best_rows,
        }
    )
    return templates.TemplateResponse(request, "overview.html", ctx)


@router.get("/data", response_class=HTMLResponse)
def market_data_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    assets = list(db.scalars(select(Asset).order_by(Asset.symbol, Asset.market_type)).all())
    datasets = list(db.scalars(select(MarketDataset).order_by(desc(MarketDataset.created_at))).all())
    jobs = list(db.scalars(select(ResearchJob).where(ResearchJob.job_type == "data_download").order_by(desc(ResearchJob.created_at)).limit(10)).all())
    ctx = _base_ctx(request, user, "data", "Market Data Center")
    ctx.update(
        {
            "assets": assets,
            "datasets": datasets,
            "timeframes": list(SUPPORTED_TIMEFRAMES),
            "download_jobs": jobs,
        }
    )
    return templates.TemplateResponse(request, "data_center.html", ctx)


@router.get("/research", response_class=HTMLResponse)
def quant_research_page(
    request: Request,
    dataset_id: int | None = None,
    autorun: int = 0,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    datasets = list(db.scalars(select(MarketDataset).order_by(desc(MarketDataset.created_at))).all())
    ctx = _base_ctx(request, user, "research", "Quant Research Center")
    ctx.update({"datasets": datasets, "selected_dataset_id": dataset_id, "autorun": autorun})
    return templates.TemplateResponse(request, "research_center.html", ctx)


@router.get("/discovery", response_class=HTMLResponse)
def discovery_page(
    request: Request,
    dataset_id: int | None = None,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    datasets = list(db.scalars(select(MarketDataset).order_by(desc(MarketDataset.created_at))).all())
    jobs = list(db.scalars(select(ResearchJob).where(ResearchJob.job_type == "strategy_discovery").order_by(desc(ResearchJob.created_at)).limit(15)).all())
    ctx = _base_ctx(request, user, "discovery", "Strategy Discovery")
    ctx.update({"datasets": datasets, "families": list(STRATEGY_FAMILIES), "jobs": jobs, "selected_dataset_id": dataset_id})
    return templates.TemplateResponse(request, "discovery.html", ctx)


@router.get("/backtesting", response_class=HTMLResponse)
def backtesting_page(
    request: Request,
    experiment_id: int | None = None,
    dataset_id: int | None = None,
    autorun: int = 0,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    datasets = list(db.scalars(select(MarketDataset).order_by(desc(MarketDataset.created_at))).all())
    exps = list(db.scalars(select(StrategyExperiment).order_by(desc(StrategyExperiment.composite_score)).limit(100)).all())
    exp_items = []
    for e in exps:
        s = db.get(StrategyDefinition, e.strategy_id)
        d = db.get(MarketDataset, e.dataset_id)
        if s and d:
            exp_items.append({"exp": e, "strat": s, "ds": d})
    ctx = _base_ctx(request, user, "backtesting", "Backtesting Center")
    ctx.update(
        {
            "datasets": datasets,
            "experiments": exp_items,
            "selected_experiment_id": experiment_id,
            "selected_dataset_id": dataset_id,
            "autorun": autorun,
        }
    )
    return templates.TemplateResponse(request, "backtesting.html", ctx)


@router.get("/validation", response_class=HTMLResponse)
def validation_page(
    request: Request,
    experiment_id: int | None = None,
    autorun: int = 0,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    exps = list(db.scalars(select(StrategyExperiment).order_by(desc(StrategyExperiment.composite_score)).limit(100)).all())
    exp_items = []
    for e in exps:
        s = db.get(StrategyDefinition, e.strategy_id)
        d = db.get(MarketDataset, e.dataset_id)
        if s and d:
            n_trials = int((e.backtest_config_json or {}).get("n_trials_attempted", 100))
            exp_items.append({"exp": e, "strat": s, "ds": d, "n_trials": n_trials})
    recent_vals_raw = list(db.scalars(select(ValidationResult).order_by(desc(ValidationResult.created_at)).limit(20)).all())
    recent_vals = []
    for v in recent_vals_raw:
        e = db.get(StrategyExperiment, v.experiment_id)
        s = db.get(StrategyDefinition, e.strategy_id) if e else None
        d = db.get(MarketDataset, e.dataset_id) if e else None
        recent_vals.append(
            {
                "val": v,
                "strategy_name": s.name if s else f"Experiment #{v.experiment_id}",
                "market": f"{d.symbol} {d.timeframe}" if d else "-",
            }
        )
    ctx = _base_ctx(request, user, "validation", "Validation Center")
    ctx.update(
        {
            "experiments": exp_items,
            "recent_validations": recent_vals,
            "selected_experiment_id": experiment_id,
            "autorun": autorun,
        }
    )
    return templates.TemplateResponse(request, "validation.html", ctx)


@router.get("/leaderboard", response_class=HTMLResponse)
def leaderboard_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    assets = sorted({d.symbol for d in db.scalars(select(MarketDataset)).all()})
    ctx = _base_ctx(request, user, "leaderboard", "Strategy Leaderboard")
    ctx.update({"symbols": assets, "timeframes": list(SUPPORTED_TIMEFRAMES), "families": list(STRATEGY_FAMILIES)})
    return templates.TemplateResponse(request, "leaderboard.html", ctx)


@router.get("/portfolio", response_class=HTMLResponse)
def portfolio_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    exps = list(db.scalars(select(StrategyExperiment).order_by(desc(StrategyExperiment.composite_score)).limit(60)).all())
    exp_items = []
    for e in exps:
        s = db.get(StrategyDefinition, e.strategy_id)
        d = db.get(MarketDataset, e.dataset_id)
        if s and d:
            exp_items.append({"exp": e, "strat": s, "ds": d})
    saved = list(db.scalars(select(PortfolioDefinition).order_by(desc(PortfolioDefinition.created_at)).limit(10)).all())
    ctx = _base_ctx(request, user, "portfolio", "Portfolio Lab")
    ctx.update({"experiments": exp_items, "saved_portfolios": saved})
    return templates.TemplateResponse(request, "portfolio.html", ctx)


@router.get("/paper-trading", response_class=HTMLResponse)
def paper_trading_page(
    request: Request,
    experiment_id: int | None = None,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    positions = list(db.scalars(select(PaperPosition).order_by(desc(PaperPosition.updated_at))).all())
    orders = list(db.scalars(select(PaperOrder).order_by(desc(PaperOrder.created_at)).limit(30)).all())
    trades = list(db.scalars(select(PaperTrade).order_by(desc(PaperTrade.created_at)).limit(30)).all())
    logs = list(db.scalars(select(SystemLog).where(SystemLog.module.like("paper_trading%")).order_by(desc(SystemLog.created_at)).limit(25)).all())
    exps = list(db.scalars(select(StrategyExperiment).order_by(desc(StrategyExperiment.composite_score)).limit(50)).all())
    exp_items = []
    for e in exps:
        s = db.get(StrategyDefinition, e.strategy_id)
        d = db.get(MarketDataset, e.dataset_id)
        if s and d:
            exp_items.append({"exp": e, "strat": s, "ds": d})

    live_cfgs = list(db.scalars(select(ExchangeConfiguration)).all())
    ctx = _base_ctx(request, user, "paper_trading", "Paper Trading Terminal")
    ctx.update(
        {
            "positions": positions,
            "orders": orders,
            "trades": trades,
            "logs": logs,
            "experiments": exp_items,
            "selected_experiment_id": experiment_id,
            "live_configs": live_cfgs,
            "global_live_flag": get_settings().live_trading_enabled,
        }
    )
    return templates.TemplateResponse(request, "paper_trading.html", ctx)


@router.get("/experiments", response_class=HTMLResponse)
def experiments_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is None:
        return _auth_redirect()
    exps = list(db.scalars(select(StrategyExperiment).order_by(desc(StrategyExperiment.created_at)).limit(120)).all())
    rows = []
    for e in exps:
        s = db.get(StrategyDefinition, e.strategy_id)
        d = db.get(MarketDataset, e.dataset_id)
        bt = db.scalar(select(BacktestResult).where(BacktestResult.experiment_id == e.id).order_by(desc(BacktestResult.id)))
        if s and d:
            rows.append({"exp": e, "strat": s, "ds": d, "bt": bt})
    ctx = _base_ctx(request, user, "experiments", "Experiment History")
    ctx.update({"rows": rows})
    return templates.TemplateResponse(request, "experiments.html", ctx)
