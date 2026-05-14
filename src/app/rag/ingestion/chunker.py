"""
Découpage du texte en chunks sémantiquement cohérents.

Pourquoi RecursiveCharacterTextSplitter ?
Il respecte la hiérarchie naturelle du texte : tente d'abord de couper
aux paragraphes (\n\n), puis aux phrases (\n), puis aux mots (espace).
Cela préserve mieux la cohérence sémantique qu'une coupure brute à position fixe.

Paramètres par défaut :
- chunk_size=1000 chars (~250 tokens) : bon équilibre contexte / précision pour RAG.
- chunk_overlap=200 : recouvrement pour éviter de couper une idée entre deux chunks.
"""

from dataclasses import dataclass

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger(__name__)


@dataclass
class TextChunk:
    """Un fragment de texte prêt à être embedé."""

    index: int
    content: str


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """
    Découpe `text` en une liste ordonnée de TextChunk.

    Retourne une liste vide si le texte est vide ou ne contient que
    des espaces — l'appelant doit gérer ce cas (422 ou skip).
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Ordre de priorité des séparateurs : du plus structurant au plus fin
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    chunks = [TextChunk(index=i, content=c) for i, c in enumerate(raw_chunks)]

    logger.info(
        "text_chunked",
        total_chunks=len(chunks),
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    return chunks
