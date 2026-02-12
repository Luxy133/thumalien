"""Suivi de la consommation énergétique du projet (Green IT)."""

import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

from codecarbon import EmissionsTracker

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/monitoring")


class EnergyTracker:
    """Suit la consommation énergétique des traitements ML/NLP."""

    def __init__(self, project_name: str = "thumalien", output_dir: Path = DEFAULT_OUTPUT_DIR):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports: list[dict] = []

    @contextmanager
    def track(self, task_name: str):
        """Context manager pour tracker la consommation d'une tâche.

        Usage:
            tracker = EnergyTracker()
            with tracker.track("classification"):
                model.predict(texts)
        """
        tracker = EmissionsTracker(
            project_name=f"{self.project_name}_{task_name}",
            output_dir=str(self.output_dir),
            log_level="warning",
        )
        tracker.start()
        start_time = datetime.now(timezone.utc)

        try:
            yield tracker
        finally:
            emissions = tracker.stop()
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            report = {
                "task": task_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "emissions_kg_co2": emissions if emissions else 0.0,
                "energy_kwh": getattr(tracker, "_total_energy", 0.0),
            }
            self.reports.append(report)
            logger.info(
                "Tâche '%s' : %.2fs, %.6f kg CO2",
                task_name, duration, report["emissions_kg_co2"],
            )

    def get_summary(self) -> dict:
        """Retourne un résumé de toutes les mesures énergétiques."""
        if not self.reports:
            return {"total_emissions_kg_co2": 0, "total_duration_seconds": 0, "tasks": []}

        return {
            "total_emissions_kg_co2": sum(r["emissions_kg_co2"] for r in self.reports),
            "total_duration_seconds": sum(r["duration_seconds"] for r in self.reports),
            "total_energy_kwh": sum(r.get("energy_kwh", 0) for r in self.reports),
            "num_tasks": len(self.reports),
            "tasks": self.reports,
        }

    def save_report(self, filename: str = "energy_report.json"):
        """Sauvegarde le rapport énergétique en JSON."""
        report_path = self.output_dir / filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        logger.info("Rapport énergétique sauvegardé dans %s", report_path)
