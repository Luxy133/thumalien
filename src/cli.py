"""CLI Thumalien — Interface en ligne de commande.

Usage :
    python -m src.cli analyze "fake news" --lang fr --limit 50
    python -m src.cli train --dataset liar --epochs 3
    python -m src.cli evaluate --model data/models/fake_news_detector
    python -m src.cli dashboard
    python -m src.cli db init
    python -m src.cli db history
"""

import os
import logging

import typer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = typer.Typer(
    name="thumalien",
    help="Thumalien — Detection de Fake News sur Bluesky",
    no_args_is_help=True,
)
db_app = typer.Typer(help="Gestion de la base de donnees")
app.add_typer(db_app, name="db")


@app.command()
def analyze(
    query: str = typer.Argument(..., help="Terme de recherche Bluesky"),
    lang: str = typer.Option(None, "--lang", "-l", help="Filtrer par langue (fr, en)"),
    limit: int = typer.Option(50, "--limit", "-n", help="Nombre de posts a analyser"),
    output: str = typer.Option("data/processed/results.json", "--output", "-o", help="Fichier de sortie JSON"),
    save_db: bool = typer.Option(False, "--save-db", help="Sauvegarder les resultats en base de donnees"),
):
    """Analyse des posts Bluesky pour detecter les fake news."""
    from src.pipeline import ThumalienPipeline

    handle = os.getenv("BLUESKY_HANDLE")
    password = os.getenv("BLUESKY_PASSWORD")
    if not handle or not password:
        typer.echo("Erreur : BLUESKY_HANDLE et BLUESKY_PASSWORD doivent etre definis dans .env", err=True)
        raise typer.Exit(1)

    typer.echo(f"Analyse de '{query}' (lang={lang}, limit={limit})...")

    pipeline = ThumalienPipeline(bluesky_handle=handle, bluesky_password=password)
    results = pipeline.run(query=query, lang=lang, limit=limit)

    if not results:
        typer.echo("Aucun post trouve.")
        raise typer.Exit(0)

    pipeline.export_results(results, output)
    typer.echo(f"{len(results)} posts analyses. Resultats dans {output}")

    # Résumé
    fiable = sum(1 for r in results if r["credibility"]["label"] == "fiable")
    douteux = sum(1 for r in results if r["credibility"]["label"] == "douteux")
    fake = sum(1 for r in results if r["credibility"]["label"] == "fake")
    typer.echo(f"  Fiables: {fiable} | Douteux: {douteux} | Fake: {fake}")

    energy = pipeline.energy_tracker.get_summary()
    typer.echo(f"  CO2: {energy['total_emissions_kg_co2']:.6f} kg | Duree: {energy['total_duration_seconds']:.1f}s")

    if save_db:
        _save_to_db(query, lang, results, energy)


@app.command()
def train(
    dataset: str = typer.Option("liar", "--dataset", "-d", help="Dataset (liar, kaggle, liar+kaggle, custom)"),
    csv: str = typer.Option(None, "--csv", help="Chemin CSV (si --dataset custom)"),
    model: str = typer.Option("roberta-base", "--model", "-m", help="Modele de base HuggingFace"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Nombre d'epoques"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Taille de batch"),
    lr: float = typer.Option(2e-5, "--lr", help="Learning rate"),
    output: str = typer.Option("data/models/fake_news_detector", "--output", "-o", help="Dossier de sortie"),
):
    """Fine-tuning du classifieur fake news."""
    from src.training.train import train as run_training

    typer.echo(f"Fine-tuning : modele={model}, dataset={dataset}, epochs={epochs}")
    metrics = run_training(
        model_name=model,
        dataset_name=dataset,
        csv_path=csv,
        output_dir=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )
    typer.echo(f"F1 macro : {metrics['test_metrics'].get('eval_f1_macro', 0):.4f}")
    typer.echo(f"Modele sauvegarde dans {output}")


@app.command()
def evaluate(
    model_path: str = typer.Option("data/models/fake_news_detector", "--model", "-m", help="Chemin du modele"),
    dataset: str = typer.Option("liar", "--dataset", "-d", help="Dataset d'evaluation"),
    csv: str = typer.Option(None, "--csv", help="Chemin CSV (si --dataset custom)"),
):
    """Evaluation detaillee du modele."""
    from src.training.evaluate import evaluate as run_eval

    typer.echo(f"Evaluation : modele={model_path}, dataset={dataset}")
    report = run_eval(model_path=model_path, dataset_name=dataset, csv_path=csv)
    typer.echo(f"Accuracy : {report['metrics']['accuracy']:.4f}")
    typer.echo(f"F1 macro : {report['metrics']['f1_macro']:.4f}")


@app.command()
def dashboard():
    """Lance le dashboard Streamlit."""
    import subprocess
    typer.echo("Lancement du dashboard sur http://localhost:8501")
    subprocess.run(["streamlit", "run", "src/dashboard/app.py", "--server.port=8501"])


@db_app.command("init")
def db_init():
    """Initialise les tables de la base de donnees."""
    from src.database.connection import get_engine
    from src.database.models import Base

    typer.echo("Initialisation de la base de donnees...")
    engine = get_engine()
    Base.metadata.create_all(engine)
    typer.echo("Tables creees avec succes.")


@db_app.command("history")
def db_history(limit: int = typer.Option(10, "--limit", "-n", help="Nombre de sessions")):
    """Affiche l'historique des sessions d'analyse."""
    from src.database.connection import get_session
    from src.database.repository import SessionRepository

    session = get_session()
    repo = SessionRepository(session)
    sessions = repo.get_history(limit=limit)

    if not sessions:
        typer.echo("Aucune session d'analyse trouvee.")
        return

    typer.echo(f"{'Date':<22s} {'Requete':<25s} {'Posts':>6s} {'Fiable':>7s} {'Douteux':>8s} {'Fake':>5s} {'CO2 (kg)':>10s}")
    typer.echo("-" * 90)
    for s in sessions:
        date = s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "?"
        typer.echo(
            f"{date:<22s} {s.query[:24]:<25s} {s.num_posts:>6d} "
            f"{s.num_fiable:>7d} {s.num_douteux:>8d} {s.num_fake:>5d} "
            f"{s.total_emissions_co2:>10.6f}"
        )
    session.close()


def _save_to_db(query, lang, results, energy):
    """Sauvegarde les résultats en BDD."""
    try:
        from src.database.connection import get_session
        from src.database.repository import save_pipeline_results

        session = get_session()
        save_pipeline_results(session, query, lang, results, energy)
        typer.echo("Resultats sauvegardes en base de donnees.")
        session.close()
    except Exception as e:
        typer.echo(f"Erreur BDD (resultats non sauvegardes) : {e}", err=True)


if __name__ == "__main__":
    app()
