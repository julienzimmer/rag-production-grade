"""
Fixtures pytest partagées.

Principes de conception :
1. La fixture `app` appelle create_app() — jamais l'instance module-level `app`.
   Cela garantit une app fraîche avec des dépendances surchargeables par test.
2. `dependency_overrides` est le mécanisme FastAPI pour injecter des settings
   de test sans modifier les fichiers .env.
3. Les fixtures sont scopées selon leur coût :
   - "session" : créé une seule fois pour toute la session de tests (DB, app)
   - "function" (défaut) : recréé pour chaque test (sessions DB, mocks)

Stratégie de test DB :
- db_engine (session) : crée les tables une fois, les supprime en fin de session
- db_session (function) : session fraîche par test, rollback après chaque test
  → isolation des données sans avoir à vider les tables manuellement
"""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Import des modèles pour les enregistrer dans Base.metadata
import app.db.models  # noqa: F401
from app.config import Settings, get_settings
from app.db.engine import Base, get_db
from app.main import create_app


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Settings spécifiques aux tests.
    Utilise une DB séparée pour ne pas polluer les données de développement.
    """
    return Settings(
        environment="development",
        debug=False,
        database_url="postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_test_db",
        log_level="DEBUG",
    )


@pytest_asyncio.fixture(scope="session")
async def db_engine(test_settings: Settings):
    """
    Crée le moteur de test et initialise le schéma DB.
    Supprime toutes les tables à la fin de la session de tests.
    """
    engine = create_async_engine(str(test_settings.database_url), echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncSession:
    """
    Session DB isolée par rollback pour chaque test.

    Pourquoi le rollback plutôt que TRUNCATE ?
    Plus rapide et sans risque d'interférence entre tests parallèles.
    Les données créées dans un test sont invisibles aux autres.
    """
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="session")
def app(test_settings: Settings):
    """
    Instance FastAPI avec les settings de test injectés via dependency_overrides.
    Session-scoped : l'app est créée une seule fois pour tous les tests.
    """
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: test_settings
    return application


@pytest.fixture
def client_with_db(app, db_session: AsyncSession):
    """
    Client HTTP de test avec la session DB injectée.
    Permet de vérifier l'état de la DB après chaque requête HTTP.
    """

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(scope="session")
def client(app) -> TestClient:
    """
    Client de test synchrone — idéal pour les tests unitaires sans DB.
    Pour les tests avec DB, utiliser client_with_db.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(autouse=True)
def disable_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Désactive LangSmith et Langfuse pour tous les tests.

    Pourquoi autouse=True ?
    Sans cette fixture, de vraies clés présentes dans l'environnement CI
    (variables d'env du runner) pourraient "fuir" dans les tests et
    déclencher de vrais appels réseau vers les services cloud.

    Langfuse : @observe devient un no-op dès que les clés sont vides.
    LangSmith : LANGCHAIN_TRACING_V2=false désactive le callback LangChain.
    OpenTelemetry : aucune action requise — get_tracer() retourne un NoOpTracer
    quand aucun TracerProvider n'est configuré (comportement par défaut OTel).
    """
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
