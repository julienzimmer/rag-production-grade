"""
Endpoint d'ingestion : POST /api/v1/ingest

Pipeline complet en une requête HTTP :
  upload → extraction texte → chunking → embeddings → persistance pgvector

Limitation actuelle : traitement synchrone dans la requête.
Pour les gros documents (> 10 Mo, > 1000 chunks), passer à une
architecture asynchrone : upload S3 + tâche Celery/SQS + webhook.
"""

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_db
from app.db.models import Chunk, Document
from app.rag.ingestion.chunker import chunk_text
from app.rag.ingestion.embedder import embed_texts
from app.rag.ingestion.loader import SUPPORTED_MIME_TYPES, load_document

router = APIRouter(tags=["ingestion"])
logger = structlog.get_logger(__name__)


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(
    file: UploadFile = File(..., description="Document PDF ou texte à ingérer"),
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Ingère un document dans la base vectorielle.

    Retourne l'identifiant du document créé et le nombre de chunks indexés.
    """
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Type MIME non supporté : {mime_type}. "
                f"Acceptés : {list(SUPPORTED_MIME_TYPES)}"
            ),
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Le fichier uploadé est vide")

    # 1. Extraction du texte
    try:
        text = load_document(content, file.filename or "unnamed", mime_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # 2. Chunking
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=422, detail="Aucun chunk extrait — document peut-être vide"
        )

    # 3. Embeddings (batch : un seul appel OpenAI pour tous les chunks)
    try:
        embeddings = await embed_texts([c.content for c in chunks])
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 4. Persistance en base de données
    document = Document(
        filename=file.filename or "unnamed",
        mime_type=mime_type,
        content=text,
        doc_metadata={"original_size_bytes": len(content)},
    )
    db.add(document)
    # flush pour obtenir document.id avant d'insérer les chunks (FK constraint)
    await db.flush()

    chunk_models = [
        Chunk(
            document_id=document.id,
            chunk_index=chunk.index,
            content=chunk.content,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    db.add_all(chunk_models)
    await db.commit()

    logger.info(
        "document_ingested",
        document_id=str(document.id),
        filename=file.filename,
        chunks=len(chunk_models),
    )

    return IngestResponse(
        document_id=str(document.id),
        filename=file.filename or "unnamed",
        chunks_created=len(chunk_models),
        message=f"Document ingéré avec succès : {len(chunk_models)} chunks créés",
    )
