"""
Moteur de base de données SQLAlchemy asynchrone.

Décisions de conception :
- create_async_engine : toutes les opérations DB sont non-bloquantes (asyncpg driver).
- async_sessionmaker avec expire_on_commit=False : évite les lazy-load
  après commit (antipattern en async — les relations expirées déclenchent
  des I/O synchrones implicites).
- get_db est une dépendance FastAPI qui yield une session par requête HTTP,
  garantissant que la session est fermée même en cas d'exception.
- _engine / _session_factory sont des singletons initialisés en lifespan
  (pas au niveau module) pour permettre de les surcharger dans les tests.
"""

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = structlog.get_logger(__name__)

# Singletons initialisés en lifespan — None jusqu'à l'appel de init_db()
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    """
    Classe de base SQLAlchemy pour tous les modèles ORM.

    Centraliser la base ici permet à Base.metadata.create_all()
    de connaître tous les modèles importés dans le projet.
    """

    pass


def init_db(
    database_url: str,
    pool_size: int = 10,
    max_overflow: int = 20,
    echo: bool = False,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """
    Initialise le pool de connexions PostgreSQL.

    Appelé une seule fois au démarrage (lifespan de FastAPI).
    Retourne l'engine pour permettre le create_all() et le dispose().
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    logger.info("db_engine_initialized", pool_size=pool_size)
    return _engine, _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dépendance FastAPI — injecte une session DB dans les handlers.

    Utilisation :
        @router.post("/ingest")
        async def ingest(db: AsyncSession = Depends(get_db)):
            ...

    La session est automatiquement fermée (et la connexion rendue au pool)
    à la fin de la requête, même en cas d'exception.
    """
    if _session_factory is None:
        raise RuntimeError("DB non initialisée — init_db() doit être appelé en lifespan")

    async with _session_factory() as session:
        yield session
