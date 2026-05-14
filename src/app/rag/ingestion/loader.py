"""
Chargement et extraction de texte depuis les documents uploadés.

Formats supportés : PDF (via PyPDF) et texte brut / markdown.
L'extraction retourne toujours une chaîne UTF-8 normalisée.
"""

import io

import pypdf
import structlog

logger = structlog.get_logger(__name__)

# Mapping MIME type → fonction d'extraction
SUPPORTED_MIME_TYPES: dict[str, str] = {
    "application/pdf": "_load_pdf",
    "text/plain": "_load_text",
    "text/markdown": "_load_text",
}


def load_document(content: bytes, filename: str, mime_type: str) -> str:
    """
    Extrait le texte brut d'un document binaire.

    Lève ValueError si le type MIME n'est pas supporté ou si aucun texte
    n'est extractible (ex. PDF scanné sans OCR).
    """
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Type MIME non supporté : {mime_type}. "
            f"Acceptés : {list(SUPPORTED_MIME_TYPES)}"
        )

    loader_fn = _LOADERS[SUPPORTED_MIME_TYPES[mime_type]]
    text = loader_fn(content, filename)

    logger.info(
        "document_loaded",
        filename=filename,
        mime_type=mime_type,
        chars=len(text),
    )
    return text


def _load_pdf(content: bytes, filename: str) -> str:
    """
    Extrait le texte de chaque page PDF avec PyPDF.

    Limitation : ne fonctionne pas sur les PDF scannés (images sans texte).
    Pour l'OCR, il faudrait ajouter pytesseract ou Amazon Textract.
    """
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())

    if not pages:
        raise ValueError(
            f"Aucun texte extractible dans {filename}. "
            "Le PDF est peut-être scanné (images sans texte)."
        )

    # Séparation par double saut de ligne pour préserver la structure des pages
    return "\n\n".join(pages)


def _load_text(content: bytes, filename: str) -> str:
    """
    Décode un fichier texte — UTF-8 en priorité, latin-1 en fallback.

    latin-1 (ISO-8859-1) ne lève jamais d'erreur de décodage car chaque
    byte 0-255 est un caractère valide. Utile pour les vieux documents Windows.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("utf8_decode_failed_fallback_latin1", filename=filename)
        return content.decode("latin-1")


# Résolution des noms de fonctions en callables (évite globals())
_LOADERS = {
    "_load_pdf": _load_pdf,
    "_load_text": _load_text,
}
