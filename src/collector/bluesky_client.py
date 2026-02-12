"""Client pour collecter les posts depuis l'API Bluesky (AT Protocol)."""

import logging
import time
from datetime import datetime, timezone

from atproto import Client
from atproto.exceptions import RequestException

logger = logging.getLogger(__name__)

# Configuration retry
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # secondes
BACKOFF_FACTOR = 2.0


class BlueskyCollector:
    """Collecte les posts publics depuis Bluesky via l'AT Protocol.

    Gère automatiquement les erreurs réseau, le rate limiting
    et le rafraîchissement de session.
    """

    def __init__(self, handle: str, password: str):
        self.handle = handle
        self.password = password
        self.client = Client()
        self._login()

    def _login(self):
        """Se connecte à Bluesky avec retry."""
        for attempt in range(MAX_RETRIES):
            try:
                self.client.login(self.handle, self.password)
                logger.info("Connecté à Bluesky en tant que %s", self.handle)
                return
            except RequestException as e:
                wait = INITIAL_BACKOFF * (BACKOFF_FACTOR ** attempt)
                logger.warning(
                    "Erreur de connexion (tentative %d/%d) : %s. Retry dans %.1fs",
                    attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
        raise ConnectionError(f"Impossible de se connecter à Bluesky après {MAX_RETRIES} tentatives")

    def _retry(self, func, *args, **kwargs):
        """Exécute une fonction avec retry et backoff exponentiel.

        Gère les erreurs réseau et le rate limiting.
        Tente un re-login si la session a expiré.
        """
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RequestException as e:
                error_msg = str(e).lower()

                # Rate limiting : attendre plus longtemps
                if "rate" in error_msg or "429" in error_msg:
                    wait = INITIAL_BACKOFF * (BACKOFF_FACTOR ** (attempt + 2))
                    logger.warning("Rate limit atteint. Attente de %.1fs...", wait)
                    time.sleep(wait)
                    continue

                # Session expirée : re-login
                if "auth" in error_msg or "401" in error_msg or "expired" in error_msg:
                    logger.warning("Session expirée, tentative de re-login...")
                    try:
                        self._login()
                        continue
                    except ConnectionError:
                        raise

                # Autre erreur : retry avec backoff
                wait = INITIAL_BACKOFF * (BACKOFF_FACTOR ** attempt)
                logger.warning(
                    "Erreur API (tentative %d/%d) : %s. Retry dans %.1fs",
                    attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)

        raise RequestException(f"Échec après {MAX_RETRIES} tentatives")

    def search_posts(self, query: str, lang: str | None = None, limit: int = 100) -> list[dict]:
        """Recherche des posts par mot-clé avec pagination et retry.

        Args:
            query: Terme de recherche.
            lang: Filtrer par langue ('fr', 'en', etc.). None = toutes.
            limit: Nombre max de posts à récupérer.

        Returns:
            Liste de posts normalisés.
        """
        posts = []
        cursor = None

        while len(posts) < limit:
            batch_size = min(limit - len(posts), 25)
            params = {"q": query, "limit": batch_size, "cursor": cursor}
            if lang:
                params["lang"] = lang

            response = self._retry(
                self.client.app.bsky.feed.search_posts, params=params
            )

            for post_view in response.posts:
                normalized = self._normalize_post(post_view)
                if normalized:
                    posts.append(normalized)

            if not response.cursor:
                break
            cursor = response.cursor

        logger.info("Collecté %d posts pour la requête '%s'", len(posts), query)
        return posts

    def get_timeline(self, limit: int = 50) -> list[dict]:
        """Récupère les posts du fil d'actualité avec retry."""
        response = self._retry(self.client.get_timeline, limit=limit)
        posts = []
        for item in response.feed:
            normalized = self._normalize_post(item.post)
            if normalized:
                posts.append(normalized)
        return posts

    def _normalize_post(self, post_view) -> dict | None:
        """Normalise un post Bluesky en dictionnaire standard.

        Retourne None si le post est invalide ou ne contient pas de texte.
        """
        try:
            record = post_view.record
            text = getattr(record, "text", None)
            if not text or not text.strip():
                return None

            return {
                "uri": post_view.uri,
                "cid": post_view.cid,
                "author_handle": post_view.author.handle,
                "author_display_name": getattr(post_view.author, "display_name", None),
                "text": text,
                "lang": getattr(record, "langs", [None]),
                "created_at": getattr(record, "created_at", None),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "like_count": getattr(post_view, "like_count", 0) or 0,
                "repost_count": getattr(post_view, "repost_count", 0) or 0,
                "reply_count": getattr(post_view, "reply_count", 0) or 0,
            }
        except Exception as e:
            logger.debug("Post ignoré (erreur de normalisation) : %s", e)
            return None
