"""Point d'entrée principal du projet Thumalien."""

import os
import logging

from dotenv import load_dotenv

from src.pipeline import ThumalienPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    load_dotenv()

    handle = os.getenv("BLUESKY_HANDLE")
    password = os.getenv("BLUESKY_PASSWORD")

    if not handle or not password:
        logger.error("BLUESKY_HANDLE et BLUESKY_PASSWORD doivent être définis dans .env")
        return

    pipeline = ThumalienPipeline(
        bluesky_handle=handle,
        bluesky_password=password,
    )

    # Exemple d'analyse
    results = pipeline.run(
        query="fake news",
        lang="fr",
        limit=50,
    )

    pipeline.export_results(results)
    logger.info("Analyse terminée : %d posts traités", len(results))


if __name__ == "__main__":
    main()
