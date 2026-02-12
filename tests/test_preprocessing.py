"""Tests unitaires pour le module de prétraitement."""

import pytest
from src.preprocessing.text_processor import clean_text, tokenize


class TestCleanText:
    def test_remove_urls(self):
        text = "Regardez cet article https://example.com/article incroyable"
        result = clean_text(text)
        assert "https://" not in result
        assert "example.com" not in result

    def test_remove_mentions(self):
        text = "Hey @user.bsky.social regarde ça"
        result = clean_text(text)
        assert "@user" not in result

    def test_normalize_whitespace(self):
        text = "trop   d'espaces    ici"
        result = clean_text(text)
        assert "  " not in result

    def test_lowercase(self):
        text = "FAKE NEWS Confirmée"
        result = clean_text(text)
        assert result == "fake news confirmée"

    def test_combined(self):
        text = "BREAKING @journaliste https://lien.com  La nouvelle est FAUSSE"
        result = clean_text(text)
        assert result == "breaking la nouvelle est fausse"


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("le chat mange la souris", lang="fr")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_filters_stopwords(self):
        tokens = tokenize("le la les un une des de du", lang="fr")
        # La plupart devraient être filtrées
        assert len(tokens) <= 2
