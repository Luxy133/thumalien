"""Détection de fake news avec score de crédibilité."""

import logging
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Labels de classification
LABELS = ["fiable", "douteux", "fake"]

# Chemin par défaut du modèle fine-tuné
FINETUNED_MODEL_PATH = Path("data/models/fake_news_detector")


class FakeNewsDetector:
    """Classifieur de fake news basé sur un modèle Transformer.

    Charge automatiquement le modèle fine-tuné s'il existe dans
    data/models/fake_news_detector/, sinon utilise le modèle de base.
    """

    def __init__(self, model_name: str = "roberta-base"):
        """Initialise le détecteur.

        Args:
            model_name: Nom du modèle HuggingFace ou chemin local.
                        Si un modèle fine-tuné existe localement, il est
                        chargé en priorité (sauf si model_name est explicitement
                        un chemin différent).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tenter de charger le modèle fine-tuné local
        resolved_model = self._resolve_model(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model, num_labels=len(LABELS)
        ).to(self.device)
        self.model.eval()
        self.is_finetuned = resolved_model != model_name or Path(model_name).exists()
        logger.info(
            "Modèle '%s' chargé sur %s (fine-tuné: %s)",
            resolved_model, self.device, self.is_finetuned,
        )

    def _resolve_model(self, model_name: str) -> str:
        """Résout le modèle à charger : fine-tuné local > modèle de base."""
        # Si l'utilisateur donne un chemin explicite, l'utiliser
        if Path(model_name).exists():
            logger.info("Chargement du modèle local : %s", model_name)
            return model_name

        # Vérifier si un modèle fine-tuné existe
        if FINETUNED_MODEL_PATH.exists() and (FINETUNED_MODEL_PATH / "config.json").exists():
            logger.info("Modèle fine-tuné détecté dans %s", FINETUNED_MODEL_PATH)
            return str(FINETUNED_MODEL_PATH)

        # Fallback sur le modèle de base
        logger.info("Pas de modèle fine-tuné trouvé, utilisation de '%s'", model_name)
        return model_name

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
