"""
Point d'entrée FastAPI.

Décision d'architecture : main.py est intentionnellement mince.
Il assemble les routers et middlewares, mais ne contient aucune logique métier.

Le pattern "app factory" (fonction create_app) rend l'app testable :
les tests appellent create_app() pour obtenir une instance fraîche avec
des dépendances injectables, sans effets de bord au niveau module.

Lancer en dev :
    uvicorn src.app.main:app --reload
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.health import router as health_router
from app.config import get_settings
from app.core.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gère le cycle de vie de l'application.

    Pourquoi lifespan plutôt que @app.on_event ?
    on_event est déprécié depuis FastAPI 0.93. Le context manager lifespan
    permet de partager des ressources (pool de connexions DB, etc.) entre
    le démarrage et l'arrêt de façon propre — sera enrichi en Phase 2
    avec l'initialisation du pool asyncpg.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level)
    logger.info(
        "application_startup",
        environment=settings.environment,
        version=settings.app_version,
    )
    yield
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        # Désactiver la doc en production évite d'exposer le schéma API publiquement
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
    )

    # CORS — doit être ajouté avant les routers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=settings.api_v1_prefix)

    return app


# Instance module — utilisée par uvicorn : uvicorn src.app.main:app
app = create_app()
