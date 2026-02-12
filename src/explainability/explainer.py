"""Explicabilité IA : identifier les mots qui influencent la classification."""

import logging

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class PredictionExplainer:
    """Explique les prédictions du classifieur en attribuant un poids à chaque mot."""

    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def explain(self, text: str, target_class: int = None) -> dict:
        """Génère une explication pour la prédiction sur un texte.

        Utilise les gradients d'attention pour estimer l'importance de chaque token.

        Args:
            text: Texte à expliquer.
            target_class: Classe cible. Si None, utilise la classe prédite.

        Returns:
            Dict avec 'tokens', 'importances', 'prediction'.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        self.model.eval()

        # Forward pass avec attention
        outputs = self.model(**inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions  # Tuple de tenseurs d'attention par couche

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1))

        # Moyenne des attentions sur toutes les couches et têtes
        # Shape: (num_layers, batch, heads, seq_len, seq_len)
        attention_weights = torch.stack(attentions).mean(dim=(0, 2))  # (batch, seq_len, seq_len)
        # Importance de chaque token = somme des attentions reçues (colonne)
        token_importances = attention_weights[0].mean(dim=0).detach().cpu().numpy()

        # Décoder les tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Filtrer les tokens spéciaux et normaliser
        word_importances = []
        for token, importance in zip(tokens, token_importances):
            if token in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>"):
                continue
            word_importances.append({
                "token": token,
                "importance": float(importance),
            })

        # Normaliser les importances
        max_imp = max(wi["importance"] for wi in word_importances) if word_importances else 1.0
        for wi in word_importances:
            wi["importance_normalized"] = wi["importance"] / max_imp if max_imp > 0 else 0.0

        # Trier par importance décroissante
        word_importances.sort(key=lambda x: x["importance"], reverse=True)

        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        return {
            "text": text,
            "predicted_class": target_class,
            "confidence": float(probabilities[target_class]),
            "top_influential_words": word_importances[:10],
            "all_word_importances": word_importances,
        }
