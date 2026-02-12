"""Client pour collecter les posts depuis l'API Bluesky (AT Protocol)."""

import logging
from datetime import datetime, timezone

from atproto import Client

logger = logging.getLogger(__name__)


class BlueskyCollector:
    """Collecte les posts publics depuis Bluesky via l'AT Protocol."""

    def __init__(self, handle: str, password: str):
        self.client = Client()
        self.client.login(handle, password)
        logger.info("Connecté à Bluesky en tant que %s", handle)

    def search_posts(self, query: str, lang: str | None = None, limit: int = 100) -> list[dict]:
        """Recherche des posts par mot-clé.

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
            response = self.client.app.bsky.feed.search_posts(
                params={"q": query, "limit": batch_size, "cursor": cursor, "lang": lang}
            )

            for post_view in response.posts:
                posts.append(self._normalize_post(post_view))

            if not response.cursor:
                break
            cursor = response.cursor

        logger.info("Collecté %d posts pour la requête '%s'", len(posts), query)
        return posts

    def get_timeline(self, limit: int = 50) -> list[dict]:
        """Récupère les posts du fil d'actualité."""
        response = self.client.get_timeline(limit=limit)
        return [self._normalize_post(item.post) for item in response.feed]

    def _normalize_post(self, post_view) -> dict:
        """Normalise un post Bluesky en dictionnaire standard."""
        record = post_view.record
        return {
            "uri": post_view.uri,
            "cid": post_view.cid,
            "author_handle": post_view.author.handle,
            "author_display_name": post_view.author.display_name,
            "text": record.text,
            "lang": getattr(record, "langs", [None]),
            "created_at": record.created_at,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "like_count": getattr(post_view, "like_count", 0),
            "repost_count": getattr(post_view, "repost_count", 0),
            "reply_count": getattr(post_view, "reply_count", 0),
        }
