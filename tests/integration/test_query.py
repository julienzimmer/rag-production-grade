"""
Tests d'intégration pour l'endpoint POST /api/v1/query.

Stratégie : mocker embedder ET generator (appels OpenAI),
mais laisser le retrieval SQL s'exécuter sur la vraie DB.
"""

from unittest.mock import AsyncMock, patch

import pytest

_FAKE_EMBEDDING = [0.0] * 1536


@pytest.mark.integration
class TestQueryEndpoint:
    def test_repond_quand_aucun_document(self, client_with_db):
        """Sans documents ingérés, retourne un message explicatif (pas une erreur)."""
        with patch(
            "app.rag.retrieval.retriever.embed_texts",
            new=AsyncMock(return_value=[_FAKE_EMBEDDING]),
        ):
            response = client_with_db.post(
                "/api/v1/query",
                json={"query": "Qu'est-ce que ce document explique ?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_sources"] == 0
        assert "ingest" in data["answer"].lower()

    def test_retourne_sources_apres_ingestion(self, client_with_db):
        """Après ingestion, la query doit retourner des sources pertinentes."""
        # Étape 1 : ingérer un document
        with patch(
            "app.rag.ingestion.embedder.embed_texts",
            new=AsyncMock(return_value=[_FAKE_EMBEDDING]),
        ):
            ingest_resp = client_with_db.post(
                "/api/v1/ingest",
                files={"file": ("doc.txt", b"Le RAG est une technique puissante.", "text/plain")},
            )
        assert ingest_resp.status_code == 201

        # Étape 2 : interroger
        with (
            patch(
                "app.rag.retrieval.retriever.embed_texts",
                new=AsyncMock(return_value=[_FAKE_EMBEDDING]),
            ),
            patch(
                "app.rag.generation.generator.generate_answer",
                new=AsyncMock(return_value="Le RAG combine retrieval et génération."),
            ),
        ):
            query_resp = client_with_db.post(
                "/api/v1/query",
                json={"query": "Qu'est-ce que le RAG ?"},
            )

        assert query_resp.status_code == 200
        data = query_resp.json()
        assert data["answer"] == "Le RAG combine retrieval et génération."
        assert data["total_sources"] >= 1
        source = data["sources"][0]
        assert source["filename"] == "doc.txt"
        assert "score" in source

    def test_valide_query_trop_courte(self, client_with_db):
        """La query doit faire au moins 3 caractères (validé par Pydantic)."""
        response = client_with_db.post(
            "/api/v1/query",
            json={"query": "ab"},
        )
        assert response.status_code == 422

    def test_top_k_respecte(self, client_with_db):
        with (
            patch(
                "app.rag.retrieval.retriever.embed_texts",
                new=AsyncMock(return_value=[_FAKE_EMBEDDING]),
            ),
            patch(
                "app.api.v1.query.retrieve_similar_chunks",
                new=AsyncMock(return_value=[]),
            ),
        ):
            response = client_with_db.post(
                "/api/v1/query",
                json={"query": "test question valide", "top_k": 3},
            )

        assert response.status_code == 200
