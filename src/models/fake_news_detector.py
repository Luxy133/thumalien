"""Détection de fake news avec score de crédibilité."""

import logging

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Labels de classification
LABELS = ["fiable", "douteux", "fake"]


class FakeNewsDetector:
    """Classifieur de fake news basé sur un modèle Transformer fine-tuné."""

    def __init__(self, model_name: str = "roberta-base"):
        """Initialise le détecteur.

        Args:
            model_name: Nom du modèle HuggingFace ou chemin local.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(LABELS)
        ).to(self.device)
        self.model.eval()
        logger.info("Modèle '%s' chargé sur %s", model_name, self.device)

    def predict(self, text: str) -> dict:
        """Prédit la crédibilité d'un texte.

        Returns:
            Dict avec 'label', 'confidence', et 'scores' par catégorie.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        predicted_idx = int(np.argmax(probabilities))

        return {
            "label": LABELS[predicted_idx],
            "confidence": float(probabilities[predicted_idx]),
            "scores": {label: float(prob) for label, prob in zip(LABELS, probabilities)},
        }

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """Prédit la crédibilité pour un lot de textes."""
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
            for probs in probabilities:
                predicted_idx = int(np.argmax(probs))
                results.append({
                    "label": LABELS[predicted_idx],
                    "confidence": float(probs[predicted_idx]),
                    "scores": {label: float(prob) for label, prob in zip(LABELS, probs)},
                })

        logger.info("Classification terminée pour %d textes", len(results))
        return results
