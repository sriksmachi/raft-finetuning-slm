"""Production-only application composition; execution APIs live separately."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.production_api import ProductionAPI
from app.store import StoreError

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def create_app(production_runs_dir: Path | None = None):
    production = ProductionAPI(production_runs_dir or ROOT / "runs" / "production")

    @asynccontextmanager
    async def lifespan(app):
        production.store.recover()
        try:
            yield
        finally:
            await production.close()

    application = FastAPI(title="SLM Studio", lifespan=lifespan)
    application.include_router(production.router)
    application.state.production = production
    application.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "app" / "static" / "index.html")

    @application.exception_handler(StoreError)
    async def store_error(request: Request, exc: StoreError):
        logger.error("Production storage error at %s: %s", request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": str(exc), "simulated": False})

    @application.middleware("http")
    async def refresh_frontend_assets(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return application


app = create_app()
