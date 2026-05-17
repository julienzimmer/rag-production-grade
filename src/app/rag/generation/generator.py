"""
Génération de la réponse finale (étape G du pipeline RAG).

Pattern : contexte récupéré → prompt augmenté → réponse générée.

Pourquoi gpt-4o-mini ?
Le contexte est déjà filtré et pertinent (retrieval step).
Un modèle léger suffit pour synthétiser des extraits pré-sélectionnés.
temperature=0 assure des réponses déterministes — important en RAG
pour la reproductibilité et les tests d'évaluation (RAGAs, Phase 4).

Intégration observabilité (Phase 3) :
- LangSmith  : actif automatiquement via LANGCHAIN_TRACING_V2=true (aucun code requis)
- Langfuse   : LangfuseCallbackHandler injecté dans ainvoke() — capture le prompt,
               la réponse, les tokens et la latence LLM dans le span @observe courant
- OpenTelemetry : span "rag.generate" avec attributs métier (modèle, nb chunks, longueur réponse)
"""

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.telemetry import get_tracer
from app.core.tracing import get_langfuse_callback_handler

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "Tu es un assistant expert qui répond aux questions uniquement à partir "
    "des extraits de documents fournis. Si les documents ne contiennent pas "
    "l'information demandée, dis-le clairement sans inventer. "
    "Cite les sources (nom du fichier) entre crochets quand pertinent."
)


async def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Génère une réponse en utilisant les chunks récupérés comme contexte.

    `context_chunks` est la liste retournée par retrieve_similar_chunks()
    — chaque dict contient au minimum 'filename' et 'content'.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY non configurée — ajoutez-la dans .env")

    # Formatage du contexte : chaque source est numérotée pour la citation
    context_text = "\n\n---\n\n".join(
        f"[Source {i + 1} : {chunk['filename']}]\n{chunk['content']}"
        for i, chunk in enumerate(context_chunks)
    )

    user_message = f"Contexte documentaire :\n\n{context_text}\n\n---\n\nQuestion : {query}"

    with get_tracer().start_as_current_span("rag.generate") as span:
        span.set_attribute("rag.context_chunks", len(context_chunks))
        span.set_attribute("rag.model", settings.openai_model)

        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0,
        )

        # Injection du LangfuseCallbackHandler : capture le prompt complet,
        # la réponse et l'usage de tokens dans la trace Langfuse en cours.
        # Injection par appel (config=) et non par constructeur : chaque invocation
        # a son propre handler isolé — safe en requêtes parallèles.
        # None si Langfuse n'est pas configuré → LangChain ignore les callbacks None.
        callbacks = [h for h in [get_langfuse_callback_handler()] if h is not None]

        response = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_message)],
            config={"callbacks": callbacks} if callbacks else {},
        )

        answer = str(response.content)
        span.set_attribute("rag.answer_length", len(answer))

    logger.info(
        "answer_generated",
        model=settings.openai_model,
        sources=len(context_chunks),
        query_preview=query[:60],
    )
    return answer
