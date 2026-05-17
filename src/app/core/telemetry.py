"""
Configuration OpenTelemetry — instrumentation HTTP et spans custom.

OpenTelemetry est le standard CNCF pour l'observabilité distribuée.
Il fonctionne en trois composants, analogues à la stack Micrometer/Zipkin de Spring :

  TracerProvider  ≈  MeterRegistry de Micrometer
  SpanProcessor   ≈  ReporterFilter (décide quand envoyer)
  Exporter        ≈  ZipkinReporter (où envoyer)

Deux niveaux d'instrumentation :
1. Auto-instrumentation FastAPI : FastAPIInstrumentor capture automatiquement
   chaque requête HTTP (method, route, status code, durée) sans modifier les endpoints.
2. Spans custom manuels : on instrumente les phases du pipeline RAG que
   l'auto-instrumentation ne connaît pas (embed, retrieve, generate).

Exporters selon l'environnement :
  development  → ConsoleSpanExporter (stdout JSON, aucune infrastructure requise)
  production   → OTLPSpanExporter (Jaeger, Grafana Tempo, Honeycomb…)
"""

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = structlog.get_logger(__name__)

# Namespace des spans custom RAG — convention : <domaine>.<opération>
# Permet de filtrer tous les spans du pipeline dans un dashboard
RAG_TRACER_NAME = "rag-pipeline"


def setup_telemetry(service_name: str, environment: str) -> None:
    """
    Configure le TracerProvider global OpenTelemetry.

    Doit être appelé une seule fois au démarrage (dans le lifespan FastAPI),
    avant tout appel à get_tracer().

    Resource : métadonnées attachées à TOUS les spans de ce processus.
    Indispensable en multi-services pour filtrer par service dans Jaeger/Tempo.

    BatchSpanProcessor vs SimpleSpanProcessor :
    - Simple  : bloquant, envoie span par span — uniquement pour les tests
    - Batch   : non-bloquant, regroupe et envoie en arrière-plan — production
    """
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            "deployment.environment": environment,
        }
    )

    provider = TracerProvider(resource=resource)

    # ConsoleSpanExporter : JSON sur stdout, lisible directement en dev
    # En production, remplacer par OTLPSpanExporter vers Grafana Tempo ou Jaeger :
    #   from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    #   exporter = OTLPSpanExporter(endpoint="http://tempo:4318/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Enregistrement global — trace.get_tracer() utilise ce provider partout
    trace.set_tracer_provider(provider)

    logger.info(
        "opentelemetry_configured",
        service=service_name,
        environment=environment,
        exporter="console",
    )


def get_tracer() -> trace.Tracer:
    """
    Retourne le tracer pour les spans custom du pipeline RAG.

    Comportement sans TracerProvider configuré (ex: tests) :
    trace.get_tracer() retourne un NoOpTracer — tous les appels sont des no-ops.
    Aucun mock nécessaire dans les tests.
    """
    return trace.get_tracer(RAG_TRACER_NAME)
