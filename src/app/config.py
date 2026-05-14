"""
Configuration de l'application via variables d'environnement.

Pourquoi Pydantic Settings ?
- Coercition de types automatique : DATABASE_URL reste une string, mais
  EMBEDDING_DIMENSION=1536 devient un int sans int() manuel.
- Validation au démarrage : si une variable requise manque, l'app plante
  immédiatement avec un message clair, pas au milieu d'une requête.
- Support .env en dev, vraies variables d'env en production
  (Docker/Kubernetes injecte directement les env vars, sans fichier .env).

Usage :
    from app.config import get_settings
    settings = get_settings()  # singleton mis en cache
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # DATABASE_URL et database_url sont équivalents
        extra="ignore",        # Variables système supplémentaires ignorées
    )

    # --- APPLICATION ---
    app_name: str = "RAG Production Grade"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- API ---
    api_v1_prefix: str = "/api/v1"
    # ALLOWED_ORIGINS="http://localhost:3000,https://monapp.com" en variable d'env
    allowed_origins: list[str] = Field(default=["http://localhost:3000"])

    # --- BASE DE DONNÉES ---
    # PostgresDsn valide le format de l'URL au démarrage
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # --- VECTOR STORE ---
    # pgvector supporte 3 métriques : cosine est optimal pour les embeddings texte
    # (la direction importe, pas la magnitude)
    vector_distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    embedding_dimension: int = 1536  # text-embedding-3-small

    # --- LLM ---
    # SecretStr empêche l'affichage accidentel des clés dans les logs
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- SUPABASE ---
    supabase_url: str | None = None
    supabase_anon_key: SecretStr | None = None
    supabase_service_role_key: SecretStr | None = None

    # --- OBSERVABILITÉ ---
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "rag-production-grade"
    langfuse_secret_key: SecretStr | None = None
    langfuse_public_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- AWS ---
    aws_region: str = "eu-west-1"
    aws_s3_bucket: str | None = None
    # Les credentials AWS ne sont PAS ici : utiliser les IAM roles en production
    # ou AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (lus nativement par boto3)


@lru_cache
def get_settings() -> Settings:
    """
    Retourne un singleton Settings mis en cache.

    lru_cache = calculé une seule fois par processus.
    Dans les tests : get_settings.cache_clear() pour réinitialiser entre les tests.
    """
    return Settings()
