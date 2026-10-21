"""Application composition; execution APIs live in separate modules."""

from contextlib import asynccontextmanager
from pathlib import Path

from app.demo_api import EVALUATION_SHOWCASE_PATH, ROOT, create_demo_app
from app.production_api import ProductionAPI


def create_app(
    runs_dir: Path | None = None,
    stage_delay: float = 0.35,
    evaluation_showcase_path: Path = EVALUATION_SHOWCASE_PATH,
    production_runs_dir: Path | None = None,
):
    application = create_demo_app(runs_dir, stage_delay, evaluation_showcase_path)
    production = ProductionAPI(production_runs_dir or ROOT / "runs" / "production")
    demo_lifespan = application.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        production.store.recover()
        async with demo_lifespan(app):
            try:
                yield
            finally:
                await production.close()

    application.router.lifespan_context = lifespan
    application.include_router(production.router)
    application.state.production = production

    @application.middleware("http")
    async def refresh_frontend_assets(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return application


app = create_app()
