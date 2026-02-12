"""Prétraitement NLP des textes collectés."""

import re
import logging

import spacy

logger = logging.getLogger(__name__)

# Chargement différé des modèles spaCy
_nlp_models: dict[str, spacy.language.Language] = {}


def get_nlp(lang: str = "fr") -> spacy.language.Language:
    """Charge et met en cache le modèle spaCy pour la langue donnée."""
    if lang not in _nlp_models:
        model_name = {"fr": "fr_core_news_sm", "en": "en_core_web_sm"}.get(lang, "fr_core_news_sm")
        _nlp_models[lang] = spacy.load(model_name)
        logger.info("Modèle spaCy '%s' chargé", model_name)
    return _nlp_models[lang]


def clean_text(text: str) -> str:
    """Nettoie le texte brut d'un post.

    - Supprime les URLs
    - Supprime les mentions (@handle)
    - Normalise les espaces
    - Convertit en minuscules
    """
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@[\w.]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize(text: str, lang: str = "fr") -> list[str]:
    """Tokenise le texte avec spaCy, en filtrant stop words et ponctuation."""
    nlp = get_nlp(lang)
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 1]


def preprocess_post(post: dict) -> dict:
    """Pipeline complet de prétraitement pour un post.

    Ajoute les champs 'clean_text' et 'tokens' au post.
    """
    lang = "fr"
    if post.get("lang"):
        langs = post["lang"]
        if isinstance(langs, list) and langs:
            lang = langs[0] if langs[0] in ("fr", "en") else "fr"

    cleaned = clean_text(post["text"])
    tokens = tokenize(cleaned, lang=lang)

    return {
        **post,
        "clean_text": cleaned,
        "tokens": tokens,
        "detected_lang": lang,
    }


def preprocess_batch(posts: list[dict]) -> list[dict]:
    """Prétraite un lot de posts."""
    processed = [preprocess_post(post) for post in posts]
    logger.info("Prétraitement terminé pour %d posts", len(processed))
    return processed
