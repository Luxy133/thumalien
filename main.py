"""Point d'entrée principal du projet Thumalien.

Usage :
    python main.py                          # Lance la CLI (aide)
    python main.py analyze "fake news"      # Analyse
    python main.py dashboard                # Dashboard
    python main.py train                    # Fine-tuning
    python main.py db init                  # Init BDD

    Ou via le module :
    python -m src analyze "fake news"
"""

from src.cli import app

if __name__ == "__main__":
    app()
