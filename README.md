# RAG Production Grade

> Portfolio AI Engineering — Projet 1/3  
> Ingénieur logiciel senior en transition vers AI Engineer.  
> Objectif : construire un système RAG complet, production-ready, étape par étape.

## Vue d'ensemble

Un système **RAG (Retrieval-Augmented Generation)** permet à un LLM de répondre à des questions en s'appuyant sur une base de documents personnalisée, plutôt que sur ses seules connaissances de pré-entraînement. Ce projet construit cette architecture de zéro, avec les outils et pratiques utilisés en production.

```
Document PDF/texte
       ↓
  [Ingestion]     → découpage en chunks → embeddings OpenAI → pgvector
       ↓
  [Retrieval]     → embedding de la question → recherche de similarité cosine
       ↓
  [Generation]    → contexte + question → LLM → réponse
       ↓
  [Évaluation]    → RAGAs / DeepEval → métriques de qualité
```

## Stack technique

| Couche | Technologie |
|---|---|
| API | Python 3.11 · FastAPI · Uvicorn |
| Validation | Pydantic v2 · pydantic-settings |
| Base de données | PostgreSQL · pgvector · SQLAlchemy async · asyncpg |
| Pipeline RAG | LangChain · LlamaIndex · OpenAI |
| Observabilité | structlog · LangSmith · Langfuse · OpenTelemetry |
| Évaluation | RAGAs · DeepEval |
| Infrastructure | Docker · GitHub Actions · Terraform · AWS (EC2, S3, ECR) |

## Structure du projet

```
rag_production_grade/
├── src/app/
│   ├── main.py              # App factory FastAPI (pattern create_app + lifespan)
│   ├── config.py            # Pydantic Settings — toute la config via .env
│   ├── api/v1/
│   │   └── health.py        # GET /api/v1/health (liveness probe Docker/K8s)
│   ├── core/
│   │   ├── logging.py       # structlog — JSON en prod, console colorée en dev
│   │   └── exceptions.py    # Exceptions HTTP personnalisées
│   └── rag/
│       ├── ingestion/       # Phase 2 — chargement, chunking, embeddings
│       ├── retrieval/       # Phase 2 — vector search, reranking
│       └── generation/      # Phase 2 — prompt, LLM, réponse
├── tests/
│   ├── conftest.py          # Fixtures pytest (TestClient, settings de test)
│   ├── unit/
│   └── integration/
├── scripts/
│   └── init_db.sql          # CREATE EXTENSION vector (auto au premier start)
├── infra/                   # Terraform — Phase 3
├── .github/workflows/
│   └── ci.yml               # Lint → Test (avec pgvector réel)
├── Dockerfile               # Multi-stage: base → builder → dev / production
├── docker-compose.yml       # postgres (pgvector) + app
├── pyproject.toml           # Dépendances groupées (core / dev / rag / eval / observability)
└── .env.example             # Toutes les variables d'environnement documentées
```

## Démarrage rapide

### Prérequis
- Docker & Docker Compose
- Python 3.11 (pour le développement local)

### Avec Docker (recommandé)

```bash
# 1. Cloner le dépôt
git clone https://github.com/julienzimmer/rag-production-grade.git
cd rag-production-grade

# 2. Configurer l'environnement
cp .env.example .env
# Éditer .env avec vos clés (OpenAI, etc.)

# 3. Lancer les services
docker-compose up

# 4. Vérifier que l'app tourne
curl http://localhost:8000/api/v1/health
# → {"status":"ok","version":"0.1.0","environment":"development"}
```

### En local (sans Docker)

```bash
# Python 3.11 requis
python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env

pytest tests/
uvicorn src.app.main:app --reload
```

## Dépendances — rôle de chaque package

### Core

**`fastapi`** — Le framework HTTP. Gère les routes, la validation des requêtes/réponses via Pydantic, l'injection de dépendances (`Depends`), et génère automatiquement la doc OpenAPI sur `/docs`.

**`uvicorn[standard]`** — Le serveur ASGI qui fait tourner FastAPI. ASGI (vs WSGI) supporte nativement l'async — indispensable pour ne pas bloquer pendant qu'on attend une réponse de l'API OpenAI. Le suffixe `[standard]` ajoute `uvloop` (boucle async en C, ~2x plus rapide) et `httptools`.

**`pydantic>=2.7`** — Moteur de validation de données, réécrit en Rust en v2 (~5x plus rapide qu'en v1). Valide automatiquement les corps de requête, query params et réponses. Si un champ attendu est `int` et que tu envoies `"abc"`, Pydantic rejette la requête avec un message clair avant que ton code soit appelé.

**`pydantic-settings>=2.3`** — Extrait de Pydantic v2 en package séparé. Charge et valide les variables d'environnement et fichiers `.env`. Permet d'écrire `DATABASE_URL=...` dans `.env` et de le retrouver typé (`PostgresDsn`) dans `Settings`.

**`asyncpg`** — Driver PostgreSQL pur-async écrit en C. C'est lui qui parle réellement à PostgreSQL — envoie les requêtes SQL et reçoit les résultats sans bloquer le thread. Utilisé par SQLAlchemy comme backend (`postgresql+asyncpg://`).

**`sqlalchemy[asyncio]>=2.0`** — L'ORM. Traduit des objets Python en SQL, gère le pool de connexions. En Phase 2, définira les tables (`documents`, `chunks`, `embeddings`) et permettra d'écrire des requêtes sans SQL brut.

**`pgvector`** — Client Python pour l'extension pgvector de PostgreSQL. Ajoute le type `Vector` à SQLAlchemy et permet les requêtes de similarité : `ORDER BY embedding <=> query_vector` (distance cosine). La brique fondamentale du RAG : stocker et chercher des embeddings directement en DB.

**`httpx`** — Client HTTP async. Utilisé par le `TestClient` de FastAPI et pour les futures intégrations (appels vers Supabase, webhooks). Plus moderne que `requests` qui est synchrone uniquement.

**`structlog`** — Logging structuré. Produit du JSON en production (parseable par CloudWatch, Grafana Loki, Datadog) et un affichage coloré lisible en développement. Permet d'attacher des champs clés-valeurs aux logs : `logger.info("embedding_created", doc_id="abc", duration_ms=142)`.

### Dev `[dev]`

**`pytest`** — Framework de tests. Découvre automatiquement les fichiers `test_*.py` et rapporte les échecs avec un diff lisible.

**`pytest-asyncio`** — Plugin pour écrire des tests `async def`. Sans lui, pytest ne sait pas exécuter du code async.

**`pytest-cov`** — Mesure la couverture de code. Génère `coverage.xml` uploadé vers Codecov dans la CI.

**`ruff`** — Linter ET formatter en Rust. Remplace `flake8` + `isort` + `black`. Instantané même sur de gros projets.

**`mypy`** — Vérificateur de types statique. Lit les annotations (`def get(id: str) -> Document`) et signale les incohérences avant l'exécution. Configuré en mode `strict`.

**`faker`** — Génère des données de test réalistes (`faker.name()`, `faker.paragraph()`). En Phase 2, crée des documents fictifs pour tester l'ingestion.

### RAG `[rag]` — Phase 2

**`langchain`** — Orchestration du pipeline RAG : enchaîne les étapes "charge → découpe → embeds → stocke → récupère → génère".

**`langchain-openai`** — Connecteur LangChain → API OpenAI (embeddings `text-embedding-3-small` + LLM `gpt-4o-mini`).

**`llama-index`** — Alternative/complément à LangChain, excellent pour l'indexation et la recherche avancée de documents.

**`openai`** — SDK officiel OpenAI. Appelé par langchain-openai en dessous.

**`tiktoken`** — Tokenizer OpenAI. Compte les tokens pour ne pas dépasser les limites de contexte lors du chunking.

**`pypdf`** — Lecture des fichiers PDF — le format de document le plus courant en entreprise.

**`python-multipart`** — Nécessaire pour que FastAPI accepte les uploads de fichiers (`UploadFile`).

### Observabilité `[observability]` — Phase 3

**`langsmith`** — Trace chaque appel LLM : inputs, outputs, latence, coût. Indispensable pour débugger les chaînes LangChain.

**`langfuse`** — Alternative open-source à LangSmith. Auto-hébergeable. Dashboards de monitoring RAG.

**`opentelemetry-sdk`** — Standard CNCF pour les traces distribuées. Envoie les spans vers Grafana Tempo ou Jaeger.

**`opentelemetry-instrumentation-fastapi`** — Instrumente automatiquement FastAPI : chaque requête HTTP devient une trace avec durée, status code, etc.

### Évaluation `[eval]` — Phase 4

**`ragas`** — Évalue la qualité du RAG sans labels humains : `faithfulness` (la réponse est-elle fidèle au contexte ?), `answer_relevancy` (répond-elle à la question ?), `context_recall` (les bons documents ont-ils été récupérés ?).

**`deepeval`** — Framework de tests unitaires pour LLMs. Permet d'écrire des assertions sur les sorties et de les intégrer dans la CI.

## Architecture des décisions techniques

### Pourquoi le pattern `create_app()` ?

Si `app` était une variable module-level, les tests importeraient l'instance avec les vraies settings. Avec `create_app()`, chaque test obtient une instance fraîche :

```python
application = create_app()
application.dependency_overrides[get_settings] = lambda: test_settings
```

### Pourquoi `lifespan` plutôt que `@app.on_event` ?

`on_event("startup")` est déprécié depuis FastAPI 0.93. Le context manager `lifespan` gère proprement le cycle démarrage/arrêt et permettra en Phase 2 d'initialiser le pool de connexions DB et de le fermer proprement.

### Pourquoi des groupes de dépendances optionnels ?

L'image Docker de production n'installe que les dépendances core. LangChain (>100 MB) n'est ajouté qu'en Phase 2. Moins de surface d'attaque, image plus légère, builds plus rapides.

### Pourquoi `condition: service_healthy` dans docker-compose ?

Sans cette condition, l'app démarre avant que PostgreSQL accepte des connexions et plante au premier accès DB. Le healthcheck `pg_isready` garantit que la DB est réellement prête.

### Pourquoi un vrai PostgreSQL dans la CI plutôt qu'un mock ?

Mocker pgvector annulerait l'intérêt de tester la logique de vector search. La CI lance un vrai conteneur `ankane/pgvector` pour détecter les régressions réelles.

### Dockerfile multi-stage — astuce cache

```dockerfile
COPY pyproject.toml ./
RUN mkdir -p src/app && touch src/app/__init__.py
RUN pip install ".[dev]"   # ← couche mise en cache
COPY src/ ./src/           # ← invalidée uniquement si le code change
```

Si tu modifies uniquement le code (pas les dépendances), `pip install` n'est pas relancé — le build prend 3s au lieu de 2min.

## Phases de développement

- [x] **Phase 1** — Scaffolding : structure, FastAPI, Docker, CI
- [ ] **Phase 2** — Pipeline RAG : ingestion PDF, chunking, embeddings, pgvector, API query
- [ ] **Phase 3** — Observabilité : LangSmith, Langfuse, OpenTelemetry
- [ ] **Phase 4** — Évaluation : RAGAs, DeepEval, métriques de qualité

## Commandes utiles

```bash
docker-compose up          # Lancer en local
pytest tests/              # Tests
ruff check .               # Lint
ruff format .              # Formatage
```

## Variables d'environnement

Voir [.env.example](.env.example) pour la liste complète et documentée.

---

*Projet réalisé dans le cadre d'un portfolio AI Engineering.*
