# RAG Production Grade

> Portfolio AI Engineering — Projet 1/3  
> Ingénieur logiciel senior en transition vers AI Engineer.  
> Objectif : construire un système RAG complet, production-ready, étape par étape.

## Vue d'ensemble

Un système **RAG (Retrieval-Augmented Generation)** permet à un LLM de répondre à des questions en s'appuyant sur une base de documents personnalisée, plutôt que sur ses seules connaissances de pré-entraînement. Ce projet construit cette architecture de zéro, avec les outils et pratiques utilisés en production.

```
Document PDF/texte
       ↓
  [Ingestion]     → extraction texte → chunking → embeddings OpenAI → pgvector
       ↓
  [Retrieval]     → embedding de la question → distance cosine → top-k chunks
       ↓
  [Generation]    → contexte + question → gpt-4o-mini → réponse + sources
       ↓
  [Évaluation]    → RAGAs / DeepEval → métriques de qualité  (Phase 4)
```

## Stack technique

| Couche | Technologie |
|---|---|
| API | Python 3.11 · FastAPI · Uvicorn |
| Validation | Pydantic v2 · pydantic-settings |
| Base de données | PostgreSQL · pgvector · SQLAlchemy async · asyncpg |
| Pipeline RAG | LangChain · LangChain-OpenAI · OpenAI SDK |
| Observabilité | structlog · LangSmith · Langfuse · OpenTelemetry |
| Évaluation | RAGAs · DeepEval |
| Infrastructure | Docker · GitHub Actions · Terraform · AWS (EC2, S3, ECR) |

## Structure du projet

```
rag_production_grade/
├── src/app/
│   ├── main.py              # App factory FastAPI + lifespan (init DB, OTel, routers)
│   ├── config.py            # Pydantic Settings — toute la config via .env
│   ├── api/v1/
│   │   ├── health.py        # GET  /api/v1/health  (liveness probe Docker/K8s)
│   │   ├── ingest.py        # POST /api/v1/ingest  (upload → chunking → pgvector)
│   │   └── query.py         # POST /api/v1/query   (retrieve + generate)
│   ├── core/
│   │   ├── logging.py       # structlog — JSON en prod, console colorée en dev
│   │   ├── tracing.py       # Langfuse v4 : observe, get_langfuse_callback_handler()
│   │   ├── telemetry.py     # OpenTelemetry : setup_telemetry(), get_tracer()
│   │   └── exceptions.py    # Exceptions HTTP personnalisées
│   ├── db/
│   │   ├── engine.py        # init_db() + get_db() — moteur SQLAlchemy async
│   │   └── models.py        # ORM : Document, Chunk (Vector 1536 dims + index HNSW)
│   └── rag/
│       ├── ingestion/
│       │   ├── loader.py    # Extraction texte : PDF (PyPDF) + texte/markdown
│       │   ├── chunker.py   # RecursiveCharacterTextSplitter (LangChain)
│       │   └── embedder.py  # OpenAI text-embedding-3-small, batch + span OTel rag.embed
│       ├── retrieval/
│       │   └── retriever.py # Recherche pgvector cosine <=> + span OTel rag.retrieve
│       └── generation/
│           └── generator.py # ChatOpenAI gpt-4o-mini + LangfuseCallbackHandler + span OTel rag.generate
├── tests/
│   ├── conftest.py          # Fixtures : test_settings, db_engine, db_session, client_with_db
│   │                        #           + fixture autouse disable_observability
│   ├── unit/
│   │   ├── test_chunker.py  # 8 tests unitaires (zéro I/O)
│   │   ├── test_tracing.py  # 3 tests — helpers Langfuse (credentials présents/absents)
│   │   └── test_telemetry.py# 4 tests — spans OTel (InMemorySpanExporter)
│   └── integration/
│       ├── test_ingest.py   # 5 tests d'intégration (DB réelle, OpenAI mocké)
│       └── test_query.py    # 4 tests d'intégration (DB réelle, OpenAI mocké)
├── scripts/
│   └── init_db.sql          # CREATE EXTENSION vector + uuid-ossp (auto au 1er start)
├── infra/                   # Terraform — Phase déploiement
├── .github/workflows/
│   └── ci.yml               # Lint → Test (avec pgvector réel)
├── Dockerfile               # Multi-stage: base → builder → dev / production
├── docker-compose.yml       # postgres (pgvector) + app (hot-reload)
├── pyproject.toml           # Dépendances groupées : core / dev / rag / eval / observability
└── .env.example             # Toutes les variables d'environnement documentées
```

## Démarrage rapide

### Prérequis
- Docker & Docker Compose
- Python 3.11 (pour le développement local)
- Clé API OpenAI (pour l'ingestion et les requêtes RAG)

### Avec Docker (recommandé)

```bash
# 1. Cloner le dépôt
git clone https://github.com/julienzimmer/rag-production-grade.git
cd rag-production-grade

# 2. Configurer l'environnement
cp .env.example .env
# Éditer .env — au minimum : OPENAI_API_KEY=sk-...

# 3. Lancer les services (postgres + app)
docker-compose up

# 4. Vérifier que l'app tourne
curl http://localhost:8000/api/v1/health
# → {"status":"ok","version":"0.1.0","environment":"development"}

# 5. Ingérer un document
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@mon_document.pdf"
# → {"document_id":"...","filename":"mon_document.pdf","chunks_created":42,"message":"..."}

# 6. Interroger la base de connaissances
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Que dit le document sur X ?", "top_k": 5}'
# → {"query":"...","answer":"...","sources":[...],"total_sources":5}
```

### En local (sans Docker)

```bash
# Python 3.11 requis
python3.11 -m venv .venv
source .venv/bin/activate

# Dépendances core + dev + rag
pip install -e ".[dev,rag]"
cp .env.example .env
# Éditer .env avec OPENAI_API_KEY et DATABASE_URL pointant vers un pgvector local

pytest tests/
uvicorn src.app.main:app --reload
```

## API — Endpoints Phase 2

### `POST /api/v1/ingest`

Upload d'un document (PDF ou texte) — le pipeline complet s'exécute dans la requête.

```
Content-Type: multipart/form-data
Body: file (PDF, text/plain, text/markdown)
```

```json
// Réponse 201
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "rapport_2024.pdf",
  "chunks_created": 87,
  "message": "Document ingéré avec succès : 87 chunks créés"
}
```

### `POST /api/v1/query`

Interrogation RAG : embed → retrieve → generate.

```json
// Corps de la requête
{
  "query": "Quels sont les points clés du rapport ?",
  "top_k": 5
}
```

```json
// Réponse 200
{
  "query": "Quels sont les points clés du rapport ?",
  "answer": "D'après le document [Source 1 : rapport_2024.pdf], les points clés sont...",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "filename": "rapport_2024.pdf",
      "content": "Extrait du document...",
      "chunk_index": 12,
      "score": 0.8742
    }
  ],
  "total_sources": 5
}
```

## Observabilité — Phase 3

Trois niveaux d'instrumentation complémentaires, chacun activable indépendamment.

### Architecture de tracing

```
Requête HTTP
     │
     ▼
[FastAPI]  ←── OpenTelemetry (span HTTP automatique : method, route, status, durée)
     │
     ├── POST /api/v1/ingest ──── @observe("rag_ingest_pipeline")  ←── trace Langfuse racine
     │       └── embed_texts()   ←── span OTel "rag.embed"
     │
     └── POST /api/v1/query ───── @observe("rag_query_pipeline")   ←── trace Langfuse racine
             ├── retrieve_similar_chunks() ←── span OTel "rag.retrieve"
             │       └── embed_texts()     ←── span OTel "rag.embed"
             └── generate_answer()         ←── span OTel "rag.generate"
                     └── ChatOpenAI.ainvoke() ←── LangSmith auto-trace
                                              ←── LangfuseCallbackHandler (tokens, latence LLM)
```

### 1. LangSmith — tracing automatique LangChain

Capture chaque appel `ChatOpenAI` : prompt complet, réponse, tokens consommés, latence. Aucune modification du code requise — LangChain détecte `LANGCHAIN_TRACING_V2=true` automatiquement.

```ini
# Dans .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...      # clé disponible sur https://smith.langchain.com
LANGCHAIN_PROJECT=rag-production-grade
```

Résultat : chaque `POST /query` crée une trace dans LangSmith avec le prompt système, la question, la réponse et l'usage de tokens.

### 2. Langfuse — tracing bout-en-bout du pipeline RAG

Trace l'ensemble du pipeline (pas seulement le LLM) : embedding, retrieval, génération, avec leurs durées imbriquées. Utilise Langfuse v4 — cloud gratuit sur [cloud.langfuse.com](https://cloud.langfuse.com).

```ini
# Dans .env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Implémentation :
- `@observe(name="rag_query_pipeline")` sur l'endpoint → trace racine
- Spans enfants automatiques sur `retrieve_similar_chunks()`, `embed_texts()`
- `LangfuseCallbackHandler` injecté dans `ChatOpenAI.ainvoke()` → span LLM avec tokens

Désactivé si les credentials sont absents (no-op transparent).

### 3. OpenTelemetry — instrumentation infrastructure

Standard CNCF. Actif en permanence, aucune configuration requise.

En développement : spans JSON affichés sur stdout (via `ConsoleSpanExporter`).  
En production : brancher un `OTLPSpanExporter` vers Grafana Tempo, Jaeger, ou Honeycomb.

Spans custom créés : `rag.embed`, `rag.retrieve`, `rag.generate` avec attributs métier (`rag.texts_count`, `rag.top_k`, `rag.results_count`, `rag.model`...).  
Auto-instrumentation FastAPI : span HTTP par requête (`http.method`, `http.route`, `http.status_code`).

### Tests d'observabilité

Les outils cloud (LangSmith, Langfuse) sont désactivés automatiquement dans les tests via une fixture `autouse` qui vide les credentials. OTel retourne un `NoOpTracer` sans provider configuré.

```bash
pytest tests/unit/test_tracing.py    # helpers Langfuse
pytest tests/unit/test_telemetry.py  # spans OTel (InMemorySpanExporter)
```

---

## Dépendances — rôle de chaque package

### Core

**`fastapi`** — Le framework HTTP. Gère les routes, la validation des requêtes/réponses via Pydantic, l'injection de dépendances (`Depends`), et génère automatiquement la doc OpenAPI sur `/docs`.

**`uvicorn[standard]`** — Le serveur ASGI qui fait tourner FastAPI. ASGI (vs WSGI) supporte nativement l'async — indispensable pour ne pas bloquer pendant qu'on attend une réponse de l'API OpenAI. Le suffixe `[standard]` ajoute `uvloop` (boucle async en C, ~2x plus rapide) et `httptools`.

**`pydantic>=2.7`** — Moteur de validation de données, réécrit en Rust en v2 (~5x plus rapide qu'en v1). Valide automatiquement les corps de requête, query params et réponses. Si un champ attendu est `int` et que tu envoies `"abc"`, Pydantic rejette la requête avec un message clair avant que ton code soit appelé.

**`pydantic-settings>=2.3`** — Extrait de Pydantic v2 en package séparé. Charge et valide les variables d'environnement et fichiers `.env`. Permet d'écrire `DATABASE_URL=...` dans `.env` et de le retrouver typé (`PostgresDsn`) dans `Settings`.

**`asyncpg`** — Driver PostgreSQL pur-async écrit en C. C'est lui qui parle réellement à PostgreSQL — envoie les requêtes SQL et reçoit les résultats sans bloquer le thread. Utilisé par SQLAlchemy comme backend (`postgresql+asyncpg://`).

**`sqlalchemy[asyncio]>=2.0`** — L'ORM. Traduit des objets Python en SQL, gère le pool de connexions. En Phase 2, définit les tables `documents` et `chunks` et permet les requêtes pgvector via les opérateurs SQLAlchemy.

**`pgvector`** — Client Python pour l'extension pgvector de PostgreSQL. Ajoute le type `Vector` à SQLAlchemy et permet les requêtes de similarité : `Chunk.embedding.cosine_distance(query_vector)`. La brique fondamentale du RAG : stocker et chercher des embeddings directement en DB.

**`httpx`** — Client HTTP async. Utilisé par le `TestClient` de FastAPI et pour les futures intégrations.

**`structlog`** — Logging structuré. Produit du JSON en production (parseable par CloudWatch, Grafana Loki, Datadog) et un affichage coloré lisible en développement. Attache des champs clés-valeurs aux logs : `logger.info("embedding_created", count=87, model="text-embedding-3-small")`.

### Dev `[dev]`

**`pytest`** — Framework de tests. Découvre automatiquement les fichiers `test_*.py` et rapporte les échecs avec un diff lisible.

**`pytest-asyncio`** — Plugin pour écrire des tests `async def`. Sans lui, pytest ne sait pas exécuter du code async. Configuré en `asyncio_mode = "auto"` : tous les tests async sont automatiquement gérés.

**`pytest-cov`** — Mesure la couverture de code. Génère `coverage.xml` uploadé vers Codecov dans la CI.

**`ruff`** — Linter ET formatter en Rust. Remplace `flake8` + `isort` + `black`. Instantané même sur de gros projets.

**`mypy`** — Vérificateur de types statique. Lit les annotations et signale les incohérences avant l'exécution. Configuré en mode `strict`.

**`faker`** — Génère des données de test réalistes. Disponible pour la Phase 4 (documents fictifs pour l'évaluation RAGAs).

### RAG `[rag]` — Phase 2

**`langchain>=0.2`** — Orchestration du pipeline RAG. Utilisé pour le text splitting (`RecursiveCharacterTextSplitter`) et comme framework LLM.

**`langchain-text-splitters>=0.2`** — Package séparé contenant les splitters (extrait du core LangChain depuis la v0.2 pour réduire les dépendances transitives). Fournit `RecursiveCharacterTextSplitter`.

**`langchain-openai>=0.1`** — Connecteur LangChain → API OpenAI. Utilisé pour `ChatOpenAI` (génération) dans `generator.py`.

**`langchain-community>=0.2`** — Intégrations communautaires LangChain (disponible pour les extensions futures).

**`openai>=1.30`** — SDK officiel OpenAI. Appelé directement dans `embedder.py` via `AsyncOpenAI` pour générer les embeddings en batch (plus de contrôle sur le batching que via LangChain).

**`tiktoken>=0.7`** — Tokenizer OpenAI. Permet de compter les tokens pour ne pas dépasser les limites de contexte.

**`pypdf>=4.2`** — Extraction de texte depuis les PDF. Utilisé dans `loader.py`. Limitation : ne supporte pas les PDF scannés (images sans texte — nécessiterait OCR).

**`python-multipart>=0.0.9`** — Nécessaire pour que FastAPI accepte les uploads de fichiers (`UploadFile` dans `POST /ingest`).

### Observabilité `[observability]` — Phase 3

**`langsmith`** — Trace chaque appel LLM : inputs, outputs, latence, coût. Indispensable pour débugger les chaînes LangChain.

**`langfuse`** — Alternative open-source à LangSmith. Auto-hébergeable. Dashboards de monitoring RAG.

**`opentelemetry-sdk`** — Standard CNCF pour les traces distribuées. Envoie les spans vers Grafana Tempo ou Jaeger.

**`opentelemetry-instrumentation-fastapi`** — Instrumente automatiquement FastAPI : chaque requête HTTP devient une trace avec durée, status code, etc.

### Évaluation `[eval]` — Phase 4

**`ragas`** — Évalue la qualité du RAG sans labels humains : `faithfulness` (la réponse est-elle fidèle au contexte ?), `answer_relevancy` (répond-elle à la question ?), `context_recall` (les bons documents ont-ils été récupérés ?).

**`deepeval`** — Framework de tests unitaires pour LLMs. Permet d'écrire des assertions sur les sorties et de les intégrer dans la CI.

## Architecture — décisions techniques

### Couche DB : engine.py et le pattern init_db()

Le moteur SQLAlchemy est initialisé dans la fonction `lifespan` de FastAPI (au démarrage), pas au niveau module. Raison : pouvoir passer une URL différente dans les tests sans modifier les fichiers de config.

```python
# main.py — lifespan
engine, _ = init_db(database_url=str(settings.database_url), ...)
async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)  # CREATE TABLE IF NOT EXISTS
```

`get_db()` est une dépendance FastAPI injectée par `Depends` dans chaque handler. Elle yield une session par requête HTTP et la ferme automatiquement (même en cas d'exception) via le context manager.

### Modèles ORM : pourquoi chunks et embeddings dans la même table ?

Un chunk a exactement un embedding (relation 1:1). Les séparer en deux tables n'apporterait qu'un JOIN inutile. L'index HNSW de pgvector opère directement sur la colonne `embedding` de la table `chunks`.

```sql
-- Index créé automatiquement par SQLAlchemy au create_all()
CREATE INDEX ix_chunks_embedding_hnsw ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

`m=16` : connexions par nœud (précision ↔ mémoire). `ef_construction=64` : qualité de l'index à la construction.

### Chunking : RecursiveCharacterTextSplitter

Le splitter tente de couper dans cet ordre : `\n\n` → `\n` → `. ` → espace → caractère. Il préserve la cohérence sémantique en respectant la structure naturelle du texte (paragraphes avant phrases avant mots). Paramètres : `chunk_size=1000` chars (~250 tokens), `chunk_overlap=200` chars.

### Embeddings : AsyncOpenAI direct (pas via LangChain)

L'API OpenAI accepte plusieurs textes en un seul appel HTTP. Appeler `embed_texts([chunk1, chunk2, ..., chunkN])` coûte un seul round-trip réseau, quelle que soit la taille du document. LangChain's `OpenAIEmbeddings` ne donne pas ce niveau de contrôle sur le batching — d'où l'appel direct au SDK `openai`.

### Distance cosine : score ∈ [0, 1]

pgvector retourne une distance cosine ∈ [0, 2]. On la convertit en score lisible :

```python
score = round(1.0 - distance / 2.0, 4)
# 0 = aucune similarité, 1 = identiques
```

### Stratégie de tests : DB réelle, OpenAI mocké

Les tests d'intégration utilisent un vrai conteneur pgvector (`rag_test_db`). L'API OpenAI est mockée avec `unittest.mock.AsyncMock` — pour ne pas consommer de tokens en CI et rendre les tests déterministes.

Chaque test dispose d'une session DB isolée par rollback : les données créées dans un test ne polluent pas les autres.

```python
@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncSession:
    async with session_factory() as session:
        yield session
        await session.rollback()  # isolation garantie
```

### Pourquoi le pattern `create_app()` ?

Si `app` était une variable module-level, les tests importeraient l'instance avec les vraies settings. Avec `create_app()`, chaque test obtient une instance fraîche :

```python
application = create_app()
application.dependency_overrides[get_settings] = lambda: test_settings
application.dependency_overrides[get_db] = override_get_db
```

### Pourquoi `condition: service_healthy` dans docker-compose ?

Sans cette condition, l'app démarre avant que PostgreSQL accepte des connexions et plante au premier accès DB. Le healthcheck `pg_isready` garantit que la DB est réellement prête.

### Dockerfile multi-stage — astuce cache

```dockerfile
COPY pyproject.toml ./
RUN mkdir -p src/app && touch src/app/__init__.py
RUN pip install ".[dev]"   # ← couche mise en cache
COPY src/ ./src/           # ← invalidée uniquement si le code change
```

Si tu modifies uniquement le code (pas les dépendances), `pip install` n'est pas relancé — le build prend 3s au lieu de 2min.

## Phases de développement

- [x] **Phase 1** — Scaffolding : FastAPI, Docker, CI, health endpoint
- [x] **Phase 2** — Pipeline RAG : ingestion PDF/texte, chunking, embeddings, pgvector, API query + generate
- [x] **Phase 3** — Observabilité : LangSmith, Langfuse v4, OpenTelemetry
- [ ] **Phase 4** — Évaluation : RAGAs, DeepEval, métriques de qualité

## Commandes utiles

```bash
# Démarrage
docker-compose up                        # Lance postgres + app

# Tests
pytest tests/                            # Tous les tests
pytest tests/unit/                       # Tests unitaires uniquement (sans DB)
pytest tests/integration/               # Tests d'intégration (nécessite pgvector)
pytest -m unit                           # Par marqueur
pytest -m integration                    # Par marqueur

# Qualité
ruff check .                             # Lint
ruff format .                            # Formatage

# Installation
pip install -e ".[dev]"                        # Dev sans RAG
pip install -e ".[dev,rag]"                   # Dev + pipeline RAG complet
pip install -e ".[dev,rag,observability]"     # Dev + RAG + observabilité (Phase 3)
```

## Variables d'environnement

Voir [.env.example](.env.example) pour la liste complète et documentée.

| Phase | Variables minimales |
|-------|---------------------|
| Phase 2 (RAG) | `DATABASE_URL` + `OPENAI_API_KEY` |
| Phase 3 (LangSmith) | + `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` |
| Phase 3 (Langfuse) | + `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` |

---

*Projet réalisé dans le cadre d'un portfolio AI Engineering.*
