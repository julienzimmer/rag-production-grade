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
- Lint : `ruff check .`
- Format : `ruff format .`
- Installer les dépendances dev : `pip install -e ".[dev]"`

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

### Phase 2 — À FAIRE : Pipeline d'ingestion RAG
- `pip install -e ".[rag]"`
- Chargement de documents (PDF, texte)
- Chunking (RecursiveCharacterTextSplitter)
- Embeddings OpenAI text-embedding-3-small (1536 dims)
- Stockage pgvector via SQLAlchemy (tables : documents, chunks, embeddings)
- Endpoints : POST /api/v1/ingest · POST /api/v1/query
- Tests pytest avec Faker

### Phase 3 — À FAIRE : Observabilité
LangSmith · Langfuse · OpenTelemetry

### Phase 4 — À FAIRE : Évaluation
RAGAs · DeepEval · métriques de qualité
