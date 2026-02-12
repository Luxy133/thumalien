"""Tests unitaires pour les modèles de classification."""

import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np


class TestFakeNewsDetector:
    def test_labels_defined(self):
        from src.models.fake_news_detector import LABELS
        assert "fiable" in LABELS
        assert "douteux" in LABELS
        assert "fake" in LABELS

    @patch("src.models.fake_news_detector.AutoModelForSequenceClassification")
    @patch("src.models.fake_news_detector.AutoTokenizer")
    def test_predict_returns_expected_format(self, mock_tokenizer_cls, mock_model_cls):
        """Vérifie le format de sortie de predict()."""
        mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        mock_model = mock_model_cls.from_pretrained.return_value
        mock_model.to.return_value = mock_model
        mock_model.return_value = MagicMock(logits=torch.tensor([[0.1, 0.8, 0.1]]))

        from src.models.fake_news_detector import FakeNewsDetector
        detector = FakeNewsDetector.__new__(FakeNewsDetector)
        detector.device = torch.device("cpu")
        detector.tokenizer = mock_tokenizer
        detector.model = mock_model

        result = detector.predict("test text")

        assert "label" in result
        assert "confidence" in result
        assert "scores" in result
        assert isinstance(result["confidence"], float)


class TestEmotionAnalyzer:
    def test_emotion_labels_defined(self):
        from src.models.emotion_analyzer import EMOTION_LABELS
        assert "colère" in EMOTION_LABELS
        assert "joie" in EMOTION_LABELS
        assert "neutre" in EMOTION_LABELS
