from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.routes.health import router as health_router
from app.api.routes.training_runs import router as training_runs_router
from app.core.lifespan import lifespan
from app.core.logging import configure_logging

configure_logging()
app = FastAPI(title="UMI GPU Training API", version="1.0.0", lifespan=lifespan)
register_error_handlers(app)
app.include_router(health_router)
app.include_router(training_runs_router)

