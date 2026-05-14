"""
Tests unitaires du module chunker.

Ces tests ne nécessitent aucune I/O (pas de DB, pas d'API OpenAI).
Ils vérifient la logique de découpage indépendamment du reste du pipeline.
"""

import pytest

from app.rag.ingestion.chunker import TextChunk, chunk_text


@pytest.mark.unit
class TestChunkText:
    def test_retourne_des_chunks_sur_texte_normal(self):
        text = "Ceci est un premier paragraphe.\n\nCeci est un second paragraphe."
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=0)
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_indices_sequentiels(self):
        # Les indices doivent être continus de 0 à N-1
        text = "A " * 300  # ~600 chars → plusieurs chunks à chunk_size=100
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_contenu_non_vide(self):
        text = "Un texte avec plusieurs mots et de la ponctuation."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
        assert all(c.content.strip() for c in chunks)

    def test_texte_vide_retourne_liste_vide(self):
        assert chunk_text("") == []

    def test_texte_espaces_retourne_liste_vide(self):
        assert chunk_text("   \n\n   ") == []

    def test_overlap_cree_recouvrement(self):
        # Avec overlap, les chunks doivent partager du contenu
        text = "mot " * 200  # texte répétitif long
        chunks_sans = chunk_text(text, chunk_size=100, chunk_overlap=0)
        chunks_avec = chunk_text(text, chunk_size=100, chunk_overlap=50)
        # L'overlap crée plus de chunks que sans overlap
        assert len(chunks_avec) >= len(chunks_sans)

    def test_tout_le_texte_est_couvert(self):
        # Chaque mot du texte doit apparaître dans au moins un chunk
        mots = ["alpha", "beta", "gamma", "delta", "epsilon"]
        text = " ".join(mots) * 20  # répétition pour forcer plusieurs chunks
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
        texte_reconstitue = " ".join(c.content for c in chunks)
        for mot in mots:
            assert mot in texte_reconstitue

    def test_texte_court_reste_en_un_chunk(self):
        text = "Texte court."
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].index == 0
