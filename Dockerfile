# Dockerfile multi-stage.
#
# Pourquoi multi-stage ?
# - Le stage "base" installe les outils système (gcc, libpq-dev) nécessaires
#   pour compiler les extensions C des packages Python (asyncpg, psycopg2).
# - Le stage "production" copie uniquement les packages compilés et le code :
#   pas de compilateur, pas d'outils de build → image plus petite = surface
#   d'attaque réduite.
# - Le stage "development" étend base avec les dépendances de dev (pytest, ruff).
#
# Construire l'image production : docker build --target production -t rag-app .
# Construire l'image dev :        docker build --target development -t rag-app-dev .

# ---- Base ----
FROM python:3.11-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement pyproject.toml d'abord — le cache Docker ne sera invalidé
# que si les dépendances changent, pas à chaque modification de code.
COPY pyproject.toml ./

# Astuce pour le cache Docker avec hatchling :
# hatchling a besoin du package src/app pour s'installer.
# On crée un stub minimal, pip installe les dépendances, puis on écrase avec le vrai code.
RUN mkdir -p src/app && touch src/app/__init__.py

# ---- Builder ----
FROM base AS builder
RUN pip install --upgrade pip && \
    pip install --no-cache-dir ".[dev]"

# ---- Development ----
FROM builder AS development
# Le vrai code source est monté en volume par docker-compose (hot-reload)
# Pas de CMD ici — fourni par docker-compose

# ---- Production ----
FROM python:3.11-slim AS production
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root pour la sécurité — ne jamais faire tourner un conteneur
# de production en tant que root
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health')"

EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
