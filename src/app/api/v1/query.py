"""
Endpoint de requête : POST /api/v1/query

Pipeline RAG complet : embed → retrieve → generate.

Pourquoi retourner les sources ?
La transparence (grounding) est essentielle en RAG production :
l'utilisateur peut vérifier les extraits qui ont fondé la réponse.
C'est aussi indispensable pour l'évaluation (RAGAs — Phase 4)
qui compare la réponse aux chunks sources.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_db
from app.rag.generation.generator import generate_answer
from app.rag.retrieval.retriever import retrieve_similar_chunks

router = APIRouter(tags=["query"])
logger = structlog.get_logger(__name__)

# Import conditionnel : si langfuse n'est pas installé, l'endpoint reste fonctionnel
try:
    # Langfuse v4 : observe et get_client dans le package racine
    from langfuse import get_client as langfuse_get_client
    from langfuse import observe as langfuse_observe

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    langfuse_get_client = None  # type: ignore[assignment]

    def langfuse_observe(name: str):  # type: ignore[misc]
        """Décorateur no-op si Langfuse n'est pas installé."""
        def decorator(func):  # type: ignore[misc]
            return func
        return decorator


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, description="Question à poser au système RAG"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre de chunks à récupérer (1–20)",
    )


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceChunk]
    total_sources: int


@router.post("/query", response_model=QueryResponse)
@langfuse_observe(name="rag_query_pipeline")
async def query_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """
    Interroge la base de connaissances RAG.

    Retourne la réponse générée et les chunks sources utilisés.
    Si aucun document n'est indexé, retourne un message explicite.

    Le décorateur @langfuse_observe crée la trace Langfuse racine.
    Les spans enfants (embed, retrieve, generate) s'y attachent automatiquement
    via la propagation de contexte asyncio (contextvars).
    """
    # Enrichir la trace Langfuse avec les métadonnées de la requête.
    # En Langfuse v4, get_client() retourne le client global configuré via env vars.
    # Si les credentials sont absents, les appels sont des no-ops silencieux.
    if _LANGFUSE_AVAILABLE and langfuse_get_client:
        client = langfuse_get_client()
        client.set_current_trace_io(input=request.query)
        client.update_current_span(metadata={"top_k": request.top_k})

    # Retrieve
    try:
        chunks = await retrieve_similar_chunks(request.query, db, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not chunks:
        return QueryResponse(
            query=request.query,
            answer=(
                "Aucun document pertinent trouvé. "
                "Ingérez d'abord des documents via POST /api/v1/ingest."
            ),
            sources=[],
            total_sources=0,
        )

    # Generate
    try:
        answer = await generate_answer(request.query, chunks)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if _LANGFUSE_AVAILABLE and langfuse_get_client:
        langfuse_get_client().update_current_span(
            output=answer,
            metadata={"chunks_used": len(chunks)},
        )

    logger.info(
        "rag_query_completed",
        query_preview=request.query[:60],
        chunks_used=len(chunks),
    )

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=[SourceChunk(**chunk) for chunk in chunks],
        total_sources=len(chunks),
    )
