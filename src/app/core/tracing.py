"""
Tracing Langfuse — observabilité du pipeline RAG.

Langfuse suit le modèle Trace → Spans, analogue aux Sleuth/Zipkin traces
en Spring Boot :
  - Une Trace  = une requête utilisateur complète (POST /query)
  - Des Spans  = les étapes imbriquées (embed → retrieve → generate)

Deux mécanismes complémentaires (API Langfuse v4) :
1. @observe (décorateur) : crée automatiquement une trace ou un span enfant
   selon le contexte d'appel. Équivalent du @NewSpan de Micrometer Tracing.
2. LangfuseCallbackHandler (langfuse.langchain) : observer LangChain qui
   capture les événements LLM (on_llm_start, on_llm_end…) et les envoie
   dans la trace @observe courante.
3. get_client() : client global Langfuse configuré via variables d'env.
   Méthodes clés : update_current_span(), set_current_trace_io(), flush().

Comportement sans credentials :
Si LANGFUSE_PUBLIC_KEY ou LANGFUSE_SECRET_KEY sont absents, Langfuse se
désactive automatiquement (log "Authentication error") — toutes les méthodes
deviennent des no-ops. Le pipeline RAG continue de fonctionner normalement.
"""

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Re-export pour que les modules consommateurs n'importent qu'ici
# (un seul point de changement si on change de lib d'observabilité)
try:
    # Langfuse v4 : observe et get_client sont dans le package racine
    from langfuse import get_client, observe  # noqa: F401
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.warning("langfuse_not_installed", hint="pip install 'langfuse>=4.0.0'")


def _is_langfuse_configured() -> bool:
    """Retourne True si les credentials Langfuse sont présents."""
    settings = get_settings()
    return bool(
        _LANGFUSE_AVAILABLE
        and settings.langfuse_secret_key
        and settings.langfuse_public_key
    )


def get_langfuse_callback_handler() -> "LangfuseCallbackHandler | None":
    """
    Retourne un LangfuseCallbackHandler prêt à être injecté dans LangChain.

    Injection recommandée dans ainvoke() plutôt qu'au constructeur ChatOpenAI :
        response = await llm.ainvoke(messages, config={"callbacks": [handler]})

    Pourquoi injecter par appel et non par constructeur ?
    Le constructeur partagerait le handler entre toutes les requêtes. En async,
    deux requêtes concurrentes contamineraient la même trace. L'injection par
    appel est request-scoped — chaque invocation a son propre handler isolé.
    """
    if not _is_langfuse_configured():
        return None

    # En Langfuse v4, le CallbackHandler utilise le client global configuré
    # via les variables d'env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY).
    # Il n'accepte plus secret_key/host en constructeur — la config est centralisée.
    return LangfuseCallbackHandler()


def flush_langfuse() -> None:
    """
    Force l'envoi des events Langfuse en attente.

    Langfuse envoie les données en batch via un thread background (non-bloquant
    pendant les requêtes). Sans flush explicite, les derniers events avant un
    arrêt propre (Ctrl+C, redéploiement K8s) pourraient être perdus.

    Appelé dans le lifespan FastAPI après le yield (phase shutdown).
    """
    if not _LANGFUSE_AVAILABLE:
        return

    try:
        from langfuse import get_client

        get_client().flush()
        logger.info("langfuse_flushed")
    except Exception:
        logger.warning("langfuse_flush_failed", exc_info=True)
