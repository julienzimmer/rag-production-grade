-- Script d'initialisation exécuté automatiquement au premier démarrage
-- du conteneur PostgreSQL (via /docker-entrypoint-initdb.d/).
--
-- Pourquoi ici plutôt que dans une migration Alembic ?
-- La création d'extensions requiert des privilèges superuser que
-- les migrations applicatives ne devraient pas avoir.
-- Ce script s'exécute en tant que l'utilisateur postgres avant que
-- l'utilisateur applicatif soit actif.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- uuid_generate_v4() pour les futurs schémas
