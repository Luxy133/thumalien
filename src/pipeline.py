"""Pipeline principal : collecte -> prétraitement -> classification -> émotions -> export."""

import logging
import json
from pathlib import Path
from datetime import datetime, timezone

from src.collector.bluesky_client import BlueskyCollector
from src.preprocessing.text_processor import preprocess_batch
from src.models.fake_news_detector import FakeNewsDetector
from src.models.emotion_analyzer import EmotionAnalyzer
from src.explainability.explainer import PredictionExplainer
from src.monitoring.energy_tracker import EnergyTracker

logger = logging.getLogger(__name__)


class ThumalienPipeline:
    """Pipeline complet de détection de fake news."""

    def __init__(
        self,
        bluesky_handle: str,
        bluesky_password: str,
        fake_news_model: str = "roberta-base",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
    ):
        self.collector = BlueskyCollector(bluesky_handle, bluesky_password)
        self.detector = FakeNewsDetector(fake_news_model)
        self.emotion_analyzer = EmotionAnalyzer(emotion_model)
        self.explainer = PredictionExplainer(self.detector.model, self.detector.tokenizer)
        self.energy_tracker = EnergyTracker()

    def run(self, query: str, lang: str | None = None, limit: int = 50) -> list[dict]:
        """Exécute le pipeline complet.

        Args:
            query: Terme de recherche Bluesky.
            lang: Filtre de langue.
            limit: Nombre de posts à analyser.

        Returns:
            Liste de posts enrichis avec scores et émotions.
        """
        logger.info("Démarrage du pipeline pour '%s'", query)

        # 1. Collecte
        with self.energy_tracker.track("collecte"):
            raw_posts = self.collector.search_posts(query, lang=lang, limit=limit)

        if not raw_posts:
            logger.warning("Aucun post trouvé pour '%s'", query)
            return []

        # 2. Prétraitement
        with self.energy_tracker.track("pretraitement"):
            processed_posts = preprocess_batch(raw_posts)

        # 3. Classification fake news
        with self.energy_tracker.track("classification"):
            texts = [p["clean_text"] for p in processed_posts]
            predictions = self.detector.predict_batch(texts)

        # 4. Analyse émotionnelle
        with self.energy_tracker.track("emotion"):
            emotions = self.emotion_analyzer.analyze_batch(texts)

        # 5. Explicabilité (top 5 posts les plus douteux)
        with self.energy_tracker.track("explicabilite"):
            enriched_posts = []
            for post, pred, emo in zip(processed_posts, predictions, emotions):
                enriched = {
                    **post,
                    "credibility": pred,
                    "emotion": emo,
                    "explanation": None,
                }

                if pred["label"] in ("douteux", "fake") and pred["confidence"] > 0.6:
                    explanation = self.explainer.explain(post["clean_text"])
                    enriched["explanation"] = {
                        "top_words": explanation["top_influential_words"][:5],
                    }

                enriched_posts.append(enriched)

        # Rapport énergétique
        self.energy_tracker.save_report()
        logger.info("Pipeline terminé : %d posts analysés", len(enriched_posts))

        return enriched_posts

    def export_results(self, results: list[dict], output_path: str = "data/processed/results.json"):
        """Exporte les résultats en JSON."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_posts": len(results),
            "results": results,
            "energy_summary": self.energy_tracker.get_summary(),
        }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Résultats exportés dans %s", output)
