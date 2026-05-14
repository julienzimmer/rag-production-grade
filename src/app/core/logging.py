"""
Configuration du logging structuré avec structlog.

Pourquoi structlog plutôt que le logging standard ?
- Produit du JSON en production (parseable par CloudWatch, Grafana Loki, Datadog).
- Produit un affichage coloré lisible en développement.
- Context binding : attacher request_id ou user_id une fois suffit — tous
  les logs suivants dans ce contexte l'incluent automatiquement.

Usage :
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info("embedding_created", doc_id="abc123", duration_ms=142)
"""

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog avec un rendu adapté à l'environnement."""

    # Redirige le logging stdlib à travers structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # JSON en production (pas de TTY), console colorée en dev
            structlog.dev.ConsoleRenderer()
            if sys.stdout.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
