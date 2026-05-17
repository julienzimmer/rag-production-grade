"""
Recherche de chunks par similarité vectorielle (pgvector).

L'opérateur pgvector <=> calcule la distance cosine entre vecteurs.
Distance cosine ∈ [0, 2] : 0 = identiques, 2 = opposés.
On la convertit en score ∈ [0, 1] via score = 1 - (distance / 2).

Pourquoi la distance cosine pour les embeddings texte ?
Les embeddings encodent la direction sémantique, pas la magnitude.
La cosine est robuste aux textes de longueurs différentes — contrairement
à la distance euclidienne (L2) qui favorise les vecteurs denses similaires.
"""

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.telemetry import get_tracer
from app.db.models import Chunk, Document
from app.rag.ingestion.embedder import embed_texts

logger = structlog.get_logger(__name__)


async def retrieve_similar_chunks(
    query: str,
    db: AsyncSession,
    top_k: int = 5,
) -> list[dict]:
    """
    Trouve les `top_k` chunks les plus proches de la requête.

    Retourne une liste de dicts triés par score décroissant (le plus
    similaire en premier). Chaque dict contient le contenu du chunk
    et les métadonnées du document source pour la citation.
    """
    with get_tracer().start_as_current_span("rag.retrieve") as span:
        span.set_attribute("rag.query_length", len(query))
        span.set_attribute("rag.top_k", top_k)

        # 1. Embedder la requête — même modèle que l'ingestion (obligatoire)
        query_embeddings = await embed_texts([query])
        query_vector = query_embeddings[0]

        # 2. Requête pgvector : distance cosine ORDER BY ASC = plus proche en premier
        stmt = (
            select(
                Chunk.id,
                Chunk.content,
                Chunk.chunk_index,
                Document.filename,
                Document.id.label("document_id"),
                Chunk.embedding.cosine_distance(query_vector).label("distance"),
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.embedding.is_not(None))
            .order_by(Chunk.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )

        result = await db.execute(stmt)
        rows = result.mappings().all()

        span.set_attribute("rag.results_count", len(rows))

    logger.info(
        "chunks_retrieved",
        query_preview=query[:60],
        top_k=top_k,
        results_found=len(rows),
    )

    return [
        {
            "chunk_id": str(row["id"]),
            "document_id": str(row["document_id"]),
            "filename": row["filename"],
            "content": row["content"],
            "chunk_index": row["chunk_index"],
            # score ∈ [0, 1] — plus lisible pour l'utilisateur final
            "score": round(1.0 - float(row["distance"]) / 2.0, 4),
        }
        for row in rows
    ]
