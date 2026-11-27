"""FastAPI main application entrypoint with persistent recovery and paper trading scheduler."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from crypto_quant_lab import __version__
from crypto_quant_lab.app.api.auth_routes import router as auth_router
from crypto_quant_lab.app.api.data_routes import router as data_router
from crypto_quant_lab.app.api.pages import router as pages_router
from crypto_quant_lab.app.api.research_routes import router as research_router
from crypto_quant_lab.app.config import get_settings
from crypto_quant_lab.core.logging import configure_logging
from crypto_quant_lab.data.downloaders.service import seed_default_assets
from crypto_quant_lab.database.session import get_session_factory, init_db
from crypto_quant_lab.paper_trading.engine import step_paper_trading_cycle
from crypto_quant_lab.workers.job_runner import recover_interrupted_jobs

logger = logging.getLogger("crypto_quant_lab.app")


async def _paper_trading_loop(stop_event: asyncio.Event) -> None:
    factory = get_session_factory()
    while not stop_event.is_set():
        try:
            await asyncio.sleep(20)
            if stop_event.is_set():
                break
            with factory() as db:
                await asyncio.to_thread(step_paper_trading_cycle, db)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("Background paper trading loop error: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    factory = get_session_factory()
    with factory() as db:
        seed_default_assets(db)
    recovered = recover_interrupted_jobs()
    if recovered:
        logger.info("Recovered %s interrupted background research jobs", recovered)

    stop_event = asyncio.Event()
    paper_task = asyncio.create_task(_paper_trading_loop(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        paper_task.cancel()
        try:
            await paper_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="CRYPTO QUANT LAB",
    version=__version__,
    description="Institutional Quantitative Cryptocurrency Research & Trading Platform",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="crypto_quant_lab/frontend/static"), name="static")
app.include_router(auth_router)
app.include_router(data_router)
app.include_router(research_router)
app.include_router(pages_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
