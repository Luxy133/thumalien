"""Analyse émotionnelle des posts (colère, peur, joie, humour, etc.)."""

import logging

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["colère", "dégoût", "peur", "joie", "tristesse", "surprise", "neutre"]


class EmotionAnalyzer:
    """Analyse les émotions dans un texte avec un modèle Transformer."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """Initialise l'analyseur d'émotions.

        Args:
            model_name: Modèle HuggingFace pour la classification d'émotions.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Mapping des labels du modèle vers nos labels français
        self._label_mapping = {
            "anger": "colère",
            "disgust": "dégoût",
            "fear": "peur",
            "joy": "joie",
            "sadness": "tristesse",
            "surprise": "surprise",
            "neutral": "neutre",
        }

        logger.info("EmotionAnalyzer chargé avec '%s'", model_name)

    def analyze(self, text: str) -> dict:
        """Analyse les émotions d'un texte.

        Returns:
            Dict avec 'dominant_emotion', 'confidence', et 'scores'.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        model_labels = self.model.config.id2label

        scores = {}
        for idx, prob in enumerate(probabilities):
            en_label = model_labels[idx]
            fr_label = self._label_mapping.get(en_label, en_label)
            scores[fr_label] = float(prob)

        dominant = max(scores, key=scores.get)

        return {
            "dominant_emotion": dominant,
            "confidence": scores[dominant],
            "scores": scores,
        }

    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """Analyse les émotions pour un lot de textes."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            model_labels = self.model.config.id2label

            for probs in probabilities:
                scores = {}
                for idx, prob in enumerate(probs):
                    en_label = model_labels[idx]
                    fr_label = self._label_mapping.get(en_label, en_label)
                    scores[fr_label] = float(prob)

                dominant = max(scores, key=scores.get)
                results.append({
                    "dominant_emotion": dominant,
                    "confidence": scores[dominant],
                    "scores": scores,
                })

        logger.info("Analyse émotionnelle terminée pour %d textes", len(results))
        return results
