"""
Endpoint de health check.

Pourquoi un endpoint dédié ?
- Le HEALTHCHECK Docker et les probes Kubernetes (liveness/readiness) appellent cet endpoint.
- Les load balancers l'utilisent pour savoir si l'instance peut recevoir du trafic.
- Un /health qui vérifie la connectivité DB est plus utile qu'un simple 200 :
  il détecte le cas "processus vivant mais DB inaccessible".

Deux niveaux de probe (pattern Kubernetes) :
- Liveness  (/health) : le processus est-il vivant ? → redémarrer si non.
- Readiness (/health/ready) : l'app peut-elle traiter des requêtes ? → retirer du
  load balancer si non. Sera implémenté en Phase 2 avec la vérification DB.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """
    Liveness probe — retourne 200 si le processus applicatif est vivant.
    Ne vérifie pas les dépendances (DB, LLM) — c'est le rôle de la readiness probe.
    """
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
    )
