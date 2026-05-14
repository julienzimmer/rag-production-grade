"""
Génération de la réponse finale (étape G du pipeline RAG).

Pattern : contexte récupéré → prompt augmenté → réponse générée.

Pourquoi gpt-4o-mini ?
Le contexte est déjà filtré et pertinent (retrieval step).
Un modèle léger suffit pour synthétiser des extraits pré-sélectionnés.
temperature=0 assure des réponses déterministes — important en RAG
pour la reproductibilité et les tests d'évaluation (RAGAs, Phase 4).
"""

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings

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

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0,
    )

    response = await llm.ainvoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )

    logger.info(
        "answer_generated",
        model=settings.openai_model,
        sources=len(context_chunks),
        query_preview=query[:60],
    )
    return str(response.content)
