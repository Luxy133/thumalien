"""Tests unitaires pour le module de training."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.training.dataset import LABEL2ID, ID2LABEL, LIAR_LABEL_MAP


class TestLabelMapping:
    def test_label2id_completeness(self):
        assert set(LABEL2ID.keys()) == {"fiable", "douteux", "fake"}

    def test_id2label_completeness(self):
        assert set(ID2LABEL.values()) == {"fiable", "douteux", "fake"}

    def test_roundtrip(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_liar_mapping_covers_all_labels(self):
        """Vérifie que tous les labels LIAR sont mappés."""
        liar_labels = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        for label in liar_labels:
            assert label in LIAR_LABEL_MAP

    def test_liar_mapping_values(self):
        """Vérifie que le mapping LIAR produit nos 3 labels."""
        mapped_values = set(LIAR_LABEL_MAP.values())
        assert mapped_values == {"fiable", "douteux", "fake"}


class TestCustomCSV:
    def test_load_custom_csv_missing_column(self, tmp_path):
        """Vérifie l'erreur si une colonne manque."""
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"wrong_col": ["text"], "label": [0]}).to_csv(csv_path, index=False)

        from src.training.dataset import load_custom_csv
        with pytest.raises(ValueError, match="Colonnes requises"):
            load_custom_csv(str(csv_path), text_column="text")

    def test_load_custom_csv_valid(self, tmp_path):
        """Vérifie le chargement d'un CSV valide."""
        csv_path = tmp_path / "test.csv"
        data = {
            "text": [f"Texte exemple numéro {i} pour tester le chargement" for i in range(100)],
            "label": [i % 3 for i in range(100)],
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        from src.training.dataset import load_custom_csv
        dataset = load_custom_csv(str(csv_path))

        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset
        assert len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"]) <= 100

    def test_load_custom_csv_text_labels(self, tmp_path):
        """Vérifie le chargement avec des labels textuels."""
        csv_path = tmp_path / "test.csv"
        data = {
            "text": [f"Texte exemple numéro {i} pour tester le chargement" for i in range(60)],
            "label": ["fiable", "douteux", "fake"] * 20,
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        from src.training.dataset import load_custom_csv
        dataset = load_custom_csv(str(csv_path))

        labels = dataset["train"]["label"]
        assert all(l in (0, 1, 2) for l in labels)


class TestDatasetStats:
    def test_get_dataset_stats(self):
        from datasets import Dataset, DatasetDict
        from src.training.dataset import get_dataset_stats

        ds = DatasetDict({
            "train": Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2]}),
        })

        stats = get_dataset_stats(ds)
        assert stats["train"]["total"] == 3
        assert "fiable" in stats["train"]["distribution"]
