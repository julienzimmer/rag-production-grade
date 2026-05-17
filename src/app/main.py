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

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.db.models  # noqa: F401  — enregistre les modèles dans Base.metadata
from app.api.v1.health import router as health_router
from app.api.v1.ingest import router as ingest_router
from app.api.v1.query import router as query_router
from app.config import get_settings
from app.core.logging import configure_logging
from app.core.telemetry import setup_telemetry
from app.core.tracing import flush_langfuse
from app.db.engine import Base, init_db

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gère le cycle de vie de l'application.

    Pourquoi lifespan plutôt que @app.on_event ?
    on_event est déprécié depuis FastAPI 0.93. Le context manager lifespan
    permet de partager des ressources entre le démarrage et l'arrêt de façon propre.

    Séquence de démarrage :
    1. Configurer les logs structurés
    2. Activer le tracing LangSmith si configuré (propagation os.environ)
    3. Initialiser le pool de connexions PostgreSQL
    4. Créer les tables (idempotent — IF NOT EXISTS)
    5. Yield (l'app est prête à servir les requêtes)
    6. Flush Langfuse + fermer le pool à l'arrêt
    """
    settings = get_settings()
    configure_logging(level=settings.log_level)
    logger.info(
        "application_startup",
        environment=settings.environment,
        version=settings.app_version,
    )

    # --- LangSmith ---
    # LangChain lit LANGCHAIN_TRACING_V2 et LANGCHAIN_API_KEY directement depuis
    # os.environ (pas depuis Pydantic Settings). On propage explicitement pour que
    # la config .env soit respectée, même dans un conteneur Docker qui injecte
    # les vars via Settings mais pas toujours dans os.environ.
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("langsmith_tracing_enabled", project=settings.langsmith_project)

    engine, _ = init_db(
        database_url=str(settings.database_url),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=settings.debug,
    )

    # create_all est idempotent (CREATE TABLE IF NOT EXISTS) — safe en dev et prod
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized")

    yield

    # --- Shutdown ---
    # Langfuse envoie les traces en batch (non-bloquant). Sans flush, les derniers
    # events avant l'arrêt (Ctrl+C, redéploiement K8s) pourraient être perdus.
    flush_langfuse()
    await engine.dispose()
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

    # --- OpenTelemetry ---
    # setup_telemetry configure le TracerProvider global (ConsoleExporter en dev).
    # instrument_app() ajoute un middleware FastAPI qui crée automatiquement un span
    # HTTP par requête (http.method, http.route, http.status_code, durée).
    # Doit être appelé AVANT add_middleware pour capturer aussi les middlewares custom.
    setup_telemetry(
        service_name=settings.app_name,
        environment=settings.environment,
    )
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        logger.warning("otel_fastapi_instrumentor_not_installed")

    # CORS — doit être ajouté avant les routers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=settings.api_v1_prefix)
    app.include_router(ingest_router, prefix=settings.api_v1_prefix)
    app.include_router(query_router, prefix=settings.api_v1_prefix)

    return app


# Instance module — utilisée par uvicorn : uvicorn src.app.main:app
app = create_app()
