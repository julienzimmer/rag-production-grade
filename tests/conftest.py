"""
Fixtures pytest partagées.

Principes de conception :
1. La fixture `app` appelle create_app() — jamais l'instance module-level `app`.
   Cela garantit une app fraîche avec des dépendances surchargeables par test.
2. `dependency_overrides` est le mécanisme FastAPI pour injecter des settings
   de test sans modifier les fichiers .env.
3. Les fixtures sont scopées selon leur coût :
   - "session" : créé une seule fois pour toute la session de tests (setup coûteux)
   - "function" (défaut) : recréé pour chaque test (isolation maximale)
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Settings spécifiques aux tests.
    Utilise une DB séparée pour ne pas polluer les données de développement.
    """
    return Settings(
        environment="development",
        debug=True,
        database_url="postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_test_db",  # noqa: E501
        log_level="DEBUG",
    )


@pytest.fixture(scope="session")
def app(test_settings: Settings):
    """
    Instance FastAPI avec les settings de test injectés via dependency_overrides.
    Session-scoped : l'app est créée une seule fois pour tous les tests.
    """
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: test_settings
    return application


@pytest.fixture(scope="session")
def client(app) -> TestClient:
    """
    Client de test synchrone — idéal pour les tests unitaires sans async.
    Pour les tests async, utiliser httpx.AsyncClient avec app= parameter.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
