"""
Exceptions HTTP personnalisées — complétées en Phase 2.

Ce fichier est intentionnellement minimal à l'initialisation.
Les exceptions métier (DocumentNotFound, EmbeddingError, etc.)
seront ajoutées quand les features correspondantes seront développées.
"""

from fastapi import HTTPException, status


class DocumentNotFoundError(HTTPException):
    def __init__(self, doc_id: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' introuvable.",
        )
