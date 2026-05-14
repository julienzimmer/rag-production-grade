"""
Tests d'intégration pour l'endpoint POST /api/v1/ingest.

Ces tests utilisent une vraie DB pgvector (rag_test_db) mais mockent
l'API OpenAI — on ne veut pas dépenser des tokens lors des tests CI.

Pourquoi mocker OpenAI et pas la DB ?
La valeur de ces tests est de vérifier l'intégration entre FastAPI,
SQLAlchemy et pgvector (les vraies requêtes SQL, les contraintes de FK,
l'insertion du vecteur). Mocker la DB supprimerait cette valeur.
"""

from unittest.mock import AsyncMock, patch

import pytest

# Embedding fictif : 1536 dimensions de zéros (valide pour pgvector)
_FAKE_EMBEDDING = [0.0] * 1536


@pytest.mark.integration
class TestIngestEndpoint:
    def test_ingere_fichier_texte(self, client_with_db):
        with patch(
            "app.rag.ingestion.embedder.embed_texts",
            new=AsyncMock(return_value=[_FAKE_EMBEDDING]),
        ):
            response = client_with_db.post(
                "/api/v1/ingest",
                files={"file": ("test.txt", b"Ceci est un document de test.", "text/plain")},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["chunks_created"] >= 1
        assert "document_id" in data

    def test_ingere_plusieurs_chunks(self, client_with_db):
        # Texte assez long pour générer plusieurs chunks
        long_text = ("Voici une phrase de test. " * 50).encode()

        with patch(
            "app.rag.ingestion.embedder.embed_texts",
            new=AsyncMock(side_effect=lambda texts: [_FAKE_EMBEDDING] * len(texts)),
        ):
            response = client_with_db.post(
                "/api/v1/ingest",
                files={"file": ("long.txt", long_text, "text/plain")},
            )

        assert response.status_code == 201
        assert response.json()["chunks_created"] >= 1

    def test_rejette_type_mime_non_supporte(self, client_with_db):
        response = client_with_db.post(
            "/api/v1/ingest",
            files={"file": ("image.png", b"\x89PNG\r\n", "image/png")},
        )
        assert response.status_code == 415

    def test_rejette_fichier_vide(self, client_with_db):
        response = client_with_db.post(
            "/api/v1/ingest",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 400

    def test_rejette_openai_key_manquante(self, client_with_db):
        with patch(
            "app.rag.ingestion.embedder.embed_texts",
            new=AsyncMock(side_effect=ValueError("OPENAI_API_KEY non configurée")),
        ):
            response = client_with_db.post(
                "/api/v1/ingest",
                files={"file": ("test.txt", b"Contenu de test valide.", "text/plain")},
            )

        assert response.status_code == 503
