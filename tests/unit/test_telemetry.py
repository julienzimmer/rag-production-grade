"""
Tests unitaires — module core/telemetry.py (OpenTelemetry).

On utilise InMemorySpanExporter pour capturer les spans en mémoire
sans infrastructure (pas de Jaeger, pas de console polluée).

Note : setup_telemetry() modifie le TracerProvider global d'OTel.
Pour isoler les tests, chaque test qui appelle setup_telemetry()
doit restaurer le provider par défaut (NoOp) en fin de test.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def in_memory_tracer():
    """
    Provider OTel en mémoire — capture les spans sans les exporter.

    SimpleSpanProcessor (au lieu de Batch) : synchrone, parfait pour les tests
    où on veut inspecter les spans immédiatement après leur fermeture.

    On ne passe pas par trace.set_tracer_provider() pour rester isolé
    du provider global utilisé par le reste de l'app pendant les tests.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


class TestGetTracer:
    def test_get_tracer_returns_noop_without_setup(self):
        """
        Sans setup_telemetry(), get_tracer() retourne un NoOpTracer.
        Les spans créés ne lèvent pas d'exception — ils sont simplement ignorés.
        """
        from app.core.telemetry import get_tracer

        tracer = get_tracer()
        # Le context manager doit fonctionner sans erreur
        with tracer.start_as_current_span("test.span") as span:
            span.set_attribute("test.key", "value")


class TestSpanAttributes:
    def test_span_captures_attributes(self, in_memory_tracer):
        """Vérifie que les attributs custom sont bien attachés aux spans."""
        tracer, exporter = in_memory_tracer

        with tracer.start_as_current_span("rag.embed") as span:
            span.set_attribute("rag.texts_count", 42)
            span.set_attribute("rag.model", "text-embedding-3-small")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "rag.embed"
        assert spans[0].attributes["rag.texts_count"] == 42
        assert spans[0].attributes["rag.model"] == "text-embedding-3-small"

    def test_nested_spans_parent_child(self, in_memory_tracer):
        """
        Vérifie que les spans imbriqués ont une relation parent-enfant correcte.
        C'est le pattern utilisé par retrieve (parent) → embed (enfant).
        """
        tracer, exporter = in_memory_tracer

        with tracer.start_as_current_span("rag.retrieve") as parent:
            parent.set_attribute("rag.top_k", 5)
            with tracer.start_as_current_span("rag.embed") as child:
                child.set_attribute("rag.texts_count", 1)

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        # Trouver parent et enfant par nom
        by_name = {s.name: s for s in spans}
        assert "rag.retrieve" in by_name
        assert "rag.embed" in by_name

        # L'enfant doit référencer le span ID du parent
        parent_ctx = by_name["rag.retrieve"].context
        child_parent_id = by_name["rag.embed"].parent
        assert child_parent_id is not None
        assert child_parent_id.span_id == parent_ctx.span_id

    def test_span_status_on_exception(self, in_memory_tracer):
        """
        Quand une exception est levée dans un span, OTel doit marquer le span ERROR.
        Ce comportement est automatique avec start_as_current_span.
        """
        from opentelemetry.trace import StatusCode

        tracer, exporter = in_memory_tracer

        with pytest.raises(ValueError):
            with tracer.start_as_current_span("rag.generate", record_exception=True):
                raise ValueError("OpenAI API error")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
