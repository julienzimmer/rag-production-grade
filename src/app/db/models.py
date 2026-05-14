"""
Modèles ORM SQLAlchemy pour le pipeline RAG.

Architecture des tables :
- documents : stocke le document source complet (texte extrait)
- chunks    : fragments de document + embedding vectoriel

Pourquoi fusionner chunk et embedding dans la même table ?
Un chunk a exactement un embedding (relation 1:1). Les séparer
ajouterait un JOIN sans bénéfice. L'index HNSW de pgvector
opère directement sur la colonne `embedding` de chunks.

Pourquoi stocker le texte brut dans documents ?
Pour pouvoir ré-chunker ou ré-embedder sans re-lire le fichier original.
En production à grand volume : déplacer vers S3 et stocker seulement l'URL.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.engine import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    # Texte brut extrait — point de re-chunking si les paramètres changent
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Métadonnées flexibles : taille originale, source S3, auteur…
    doc_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Vector(1536) = text-embedding-3-small — nullable le temps de l'ingestion
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        # Index HNSW pour la recherche de plus proches voisins approximative.
        # Pourquoi HNSW plutôt qu'IVFFlat ?
        # HNSW (Hierarchical Navigable Small World) est plus rapide en requête
        # et ne nécessite pas de données pré-existantes pour calibrer les clusters.
        # IVFFlat est plus rapide à construire mais requiert des listes pré-calculées.
        # m=16 : nombre de connexions par nœud (précision vs mémoire).
        # ef_construction=64 : taille de la liste de candidats à la construction (qualité de l'index).
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
