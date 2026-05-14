"""
Génération d'embeddings vectoriels via l'API OpenAI.

Modèle : text-embedding-3-small (1536 dimensions)
- Meilleur rapport qualité/coût vs text-embedding-3-large (3072 dims)
- Coût ~$0.02 / million tokens
- Performances MTEB suffisantes pour la quasi-totalité des cas RAG

Pourquoi traiter en batch ?
L'API OpenAI accepte plusieurs textes en un seul appel HTTP, ce qui
réduit la latence réseau et le nombre de requêtes (important à l'ingestion
de gros documents avec des centaines de chunks).
"""

import structlog
from openai import AsyncOpenAI

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Limite de l'API OpenAI : 2048 inputs par requête d'embeddings
_BATCH_SIZE = 100  # conservateur pour éviter les timeouts sur textes longs


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Génère les embeddings pour une liste de textes.

    Traite en sous-batches si nécessaire. Retourne les embeddings
    dans le même ordre que `texts`.

    Lève ValueError si OPENAI_API_KEY n'est pas configurée.
    Lève openai.APIError en cas d'erreur API (propagée vers l'appelant).
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY non configurée — ajoutez-la dans .env")

    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        # Les embeddings sont retournés dans l'ordre des inputs
        all_embeddings.extend(item.embedding for item in response.data)

    logger.info(
        "embeddings_created",
        count=len(all_embeddings),
        model=settings.openai_embedding_model,
    )
    return all_embeddings
