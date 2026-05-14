# CLAUDE.md

## Stack
Python 3.11 · FastAPI · LangChain · LlamaIndex · pgvector
Supabase · RAGAs · DeepEval · LangSmith · Langfuse
Docker · GitHub Actions · Terraform · AWS (EC2, S3, ECR)

## Conventions
- Toujours expliquer les choix techniques importants
- Commenter le code de façon pédagogique
- Procéder étape par étape, une fonctionnalité à la fois
- Tests pytest sur les composants critiques
- Variables d'environnement via .env (jamais en dur)

## Commandes utiles
- Lancer en local : `docker-compose up`
- Tests : `pytest tests/`
- Tests unitaires seuls (sans DB) : `pytest tests/unit/`
- Tests d'intégration : `pytest tests/integration/`
- Lint : `ruff check .`
- Format : `ruff format .`
- Installer dev + RAG : `pip install -e ".[dev,rag]"`

## Repo GitHub
https://github.com/julienzimmer/rag-production-grade

## Environnement machine
- Linux Mint 20.1 (Ubuntu 20.04), x86_64
- Python 3.11 à installer (deadsnakes PPA ou pyenv) — 3.8 système, 3.12 disponible mais _sqlite3 manquant
- git configuré : user julienzimmer, email julien.zimmer83@gmail.com

## Roadmap

### Phase 1 — COMPLÈTE
Scaffolding production-grade :
- App factory FastAPI + Pydantic Settings v2 + structlog
- GET /api/v1/health (liveness probe)
- Docker multi-stage + docker-compose avec pgvector
- pyproject.toml avec groupes optionnels (dev / rag / eval / observability)
- CI GitHub Actions : lint → test
- README complet

### Phase 2 — COMPLÈTE : Pipeline RAG

Pipeline complet d'ingestion et de requête RAG.

#### Fichiers créés

**Couche DB** (`src/app/db/`)
- `engine.py` — moteur SQLAlchemy async
  - `init_db(database_url, pool_size, max_overflow, echo)` : initialise le singleton engine + session_factory, appelé dans le lifespan FastAPI
  - `get_db()` : dépendance FastAPI, yield une `AsyncSession` par requête HTTP
  - `Base` : classe déclarative SQLAlchemy (tous les modèles en héritent)
- `models.py` — modèles ORM
  - `Document` : table `documents` — id (UUID PK), filename, mime_type, content (Text), doc_metadata (JSONB), created_at
  - `Chunk` : table `chunks` — id (UUID PK), document_id (FK), chunk_index, content (Text), embedding (Vector 1536), token_count, created_at
  - Index HNSW cosine sur `chunks.embedding` (m=16, ef_construction=64)

**Pipeline d'ingestion** (`src/app/rag/ingestion/`)
- `loader.py`
  - `load_document(content, filename, mime_type) -> str` : dispatch vers `_load_pdf` ou `_load_text`
  - `_load_pdf` : extraction page par page avec PyPDF, lève ValueError si aucun texte (PDF scanné)
  - `_load_text` : décodage UTF-8 avec fallback latin-1
  - `SUPPORTED_MIME_TYPES` : dict utilisé par l'endpoint pour valider le type MIME (415 si non supporté)
- `chunker.py`
  - `chunk_text(text, chunk_size=1000, chunk_overlap=200) -> list[TextChunk]` : découpage avec `RecursiveCharacterTextSplitter` de `langchain_text_splitters`
  - `TextChunk` : dataclass `{index: int, content: str}`
  - Séparateurs dans l'ordre : `\n\n`, `\n`, `. `, espace, caractère
  - Retourne `[]` si le texte est vide/blank
- `embedder.py`
  - `embed_texts(texts: list[str]) -> list[list[float]]` : appel `AsyncOpenAI.embeddings.create` en batch (100 textes max par appel)
  - Modèle : `text-embedding-3-small` (1536 dimensions), lu depuis `settings.openai_embedding_model`
  - Lève `ValueError` si `OPENAI_API_KEY` est absente

**Retrieval** (`src/app/rag/retrieval/`)
- `retriever.py`
  - `retrieve_similar_chunks(query, db, top_k=5) -> list[dict]` : embed la query puis requête pgvector
  - Opérateur pgvector `<=>` (distance cosine), ORDER BY ASC
  - Score retourné : `1.0 - distance / 2.0` (normalisation ∈ [0, 1])
  - Chaque résultat : `{chunk_id, document_id, filename, content, chunk_index, score}`

**Generation** (`src/app/rag/generation/`)
- `generator.py`
  - `generate_answer(query, context_chunks) -> str` : ChatOpenAI gpt-4o-mini, temperature=0
  - Prompt système fixe (répondre uniquement d'après le contexte, citer les sources)
  - Contexte formaté avec numérotation `[Source N : filename]`

**Endpoints** (`src/app/api/v1/`)
- `ingest.py` — `POST /api/v1/ingest`
  - Entrée : `multipart/form-data`, champ `file`
  - Validation : MIME type (415), fichier vide (400), texte vide après extraction (422), OpenAI KO (503)
  - Séquence : `load_document → chunk_text → embed_texts → flush(Document) → add_all(Chunks) → commit`
  - Réponse 201 : `{document_id, filename, chunks_created, message}`
- `query.py` — `POST /api/v1/query`
  - Entrée JSON : `{query: str (min 3 chars), top_k: int (1–20, défaut 5)}`
  - Si aucun chunk trouvé : 200 avec message explicatif (pas d'erreur)
  - Réponse 200 : `{query, answer, sources: [SourceChunk], total_sources}`

#### Fichiers modifiés

- `src/app/main.py` — ajout import `app.db.models` (enregistre les modèles dans `Base.metadata`), `init_db()` + `Base.metadata.create_all()` dans le lifespan, inclusion des routers `ingest` et `query`
- `pyproject.toml` — ajout `langchain-text-splitters>=0.2.0` dans le groupe `rag`
- `tests/conftest.py` — ajout fixtures DB : `db_engine` (session, crée/détruit les tables), `db_session` (function, rollback après chaque test), `client_with_db` (override `get_db` pour injecter la session de test)

#### Tests

- `tests/unit/test_chunker.py` — 8 tests unitaires, zéro I/O
  - Cas : texte normal, indices séquentiels, contenu non vide, texte vide, espaces seuls, effet de l'overlap, couverture complète, texte court = 1 chunk
- `tests/integration/test_ingest.py` — 5 tests (DB pgvector réelle, OpenAI mocké avec `AsyncMock`)
  - Cas : ingestion TXT, ingestion avec plusieurs chunks, type MIME invalide (415), fichier vide (400), OpenAI KO (503)
- `tests/integration/test_query.py` — 4 tests (DB pgvector réelle, OpenAI mocké)
  - Cas : 0 document indexé, sources retournées après ingestion, query trop courte (422), top_k respecté

#### Stratégie DB dans les tests

La fixture `db_engine` (scope=session) crée les tables en début de session et les supprime à la fin. La fixture `db_session` (scope=function) rollback après chaque test → isolation complète des données sans TRUNCATE.

### Phase 3 — À FAIRE : Observabilité
LangSmith · Langfuse · OpenTelemetry

### Phase 4 — À FAIRE : Évaluation
RAGAs · DeepEval · métriques de qualité
