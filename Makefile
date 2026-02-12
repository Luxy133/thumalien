.PHONY: help install dev lint format test test-cov run dashboard train evaluate db-init docker clean

help: ## Afficher cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# === Installation ===

install: ## Installer les dependances
	pip install -r requirements.txt
	python -m spacy download fr_core_news_sm
	python -m spacy download en_core_web_sm

dev: install ## Installer les dependances + outils de dev
	pip install ruff pre-commit
	pre-commit install

# === Qualite de code ===

lint: ## Verifier le code avec ruff
	ruff check src/ tests/

format: ## Formater le code avec ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

# === Tests ===

test: ## Lancer les tests
	pytest tests/ -v --tb=short

test-cov: ## Lancer les tests avec couverture
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Rapport HTML : htmlcov/index.html"

# === Application ===

run: ## Lancer l'analyse (defaut: "fake news", 50 posts FR)
	python -m src.cli analyze "fake news" --lang fr --limit 50

dashboard: ## Lancer le dashboard Streamlit
	streamlit run src/dashboard/app.py --server.port=8501

# === Training ===

train: ## Fine-tuner le modele sur LIAR
	python -m src.cli train --dataset liar --epochs 3

evaluate: ## Evaluer le modele fine-tune
	python -m src.cli evaluate --model data/models/fake_news_detector

# === Base de donnees ===

db-init: ## Initialiser les tables PostgreSQL
	python -m src.cli db init

# === Docker ===

docker: ## Lancer avec Docker Compose
	docker-compose up --build

docker-down: ## Arreter Docker Compose
	docker-compose down

# === Nettoyage ===

clean: ## Nettoyer les fichiers temporaires
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f coverage.xml .coverage
