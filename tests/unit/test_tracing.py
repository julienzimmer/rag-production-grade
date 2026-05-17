"""
Tests unitaires — module core/tracing.py (Langfuse).

Stratégie : on ne teste pas Langfuse lui-même (bibliothèque externe),
mais le comportement de nos helpers selon que les credentials sont
présents ou absents.
"""

import pytest


class TestGetLangfuseCallbackHandler:
    def test_returns_none_without_credentials(self):
        """Sans credentials, le handler doit être None (pas d'appel réseau)."""
        # La fixture autouse disable_observability vide les credentials
        # dans os.environ, mais get_settings() est mis en cache via lru_cache.
        # On invalide le cache pour que Settings recharge les vars d'env patchées.
        from app.config import get_settings
        from app.core.tracing import get_langfuse_callback_handler

        get_settings.cache_clear()
        handler = get_langfuse_callback_handler()
        assert handler is None

    def test_flush_langfuse_without_credentials_does_not_raise(self):
        """flush_langfuse() doit être silencieux si Langfuse n'est pas configuré."""
        from app.config import get_settings
        from app.core.tracing import flush_langfuse

        get_settings.cache_clear()
        # Ne doit pas lever d'exception
        flush_langfuse()

    def test_langfuse_configured_returns_handler(self, monkeypatch: pytest.MonkeyPatch):
        """Avec de vrais credentials (fakés), le handler doit être instancié."""
        # Vérifier d'abord que langfuse.langchain est disponible (v4+)
        try:
            from langfuse.langchain import CallbackHandler  # noqa: F401
        except ImportError:
            pytest.skip("langfuse.langchain non disponible")

        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-fake-secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-fake-public")

        from app.config import get_settings
        from app.core.tracing import get_langfuse_callback_handler

        get_settings.cache_clear()
        try:
            handler = get_langfuse_callback_handler()
            # L'instanciation du handler ne fait pas d'appel réseau immédiat
            assert handler is not None
        finally:
            get_settings.cache_clear()
