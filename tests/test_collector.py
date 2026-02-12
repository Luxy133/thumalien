"""Tests unitaires pour le collecteur Bluesky."""

import pytest
from unittest.mock import MagicMock, patch
from src.collector.bluesky_client import BlueskyCollector


class TestBlueskyCollector:
    @patch("src.collector.bluesky_client.Client")
    def test_normalize_post(self, mock_client_class):
        """Vérifie que _normalize_post retourne le bon format."""
        mock_client_class.return_value.login = MagicMock()

        collector = BlueskyCollector("test.bsky.social", "password")

        mock_post = MagicMock()
        mock_post.uri = "at://did:plc:xxx/app.bsky.feed.post/yyy"
        mock_post.cid = "bafyreiabc123"
        mock_post.author.handle = "user.bsky.social"
        mock_post.author.display_name = "Test User"
        mock_post.record.text = "Ceci est un test"
        mock_post.record.langs = ["fr"]
        mock_post.record.created_at = "2025-01-01T00:00:00Z"
        mock_post.like_count = 5
        mock_post.repost_count = 2
        mock_post.reply_count = 1

        result = collector._normalize_post(mock_post)

        assert result["text"] == "Ceci est un test"
        assert result["author_handle"] == "user.bsky.social"
        assert result["uri"] == mock_post.uri
        assert "collected_at" in result

    @patch("src.collector.bluesky_client.Client")
    def test_search_posts_empty(self, mock_client_class):
        """Vérifie le comportement avec 0 résultats."""
        mock_client = mock_client_class.return_value
        mock_client.login = MagicMock()

        mock_response = MagicMock()
        mock_response.posts = []
        mock_response.cursor = None
        mock_client.app.bsky.feed.search_posts.return_value = mock_response

        collector = BlueskyCollector("test.bsky.social", "password")
        results = collector.search_posts("test", limit=10)

        assert results == []
