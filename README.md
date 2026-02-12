# Thumalien -Detction de Fake News sur Bluesky

> Pipeline NLP de detection automatisee des fake news sur le reseau social Bluesky, avec analyse emotionnelle, explicabilite IA et suivi energetique.

**Projet d'etude - Master 1 Data & IA**

---

## Table des matieres

- [Presentation](#presentation)
- [Fonctionnalites](#fonctionnalites)
- [Architecture](#architecture)
- [Installation](#installation)
- [Demarrage rapide](#demarrage-rapide)
- [Utilisation](#utilisation)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Tests](#tests)
- [Documentation](#documentation)
- [Contribution](#contribution)

---

## Presentation

Les reseaux sociaux sont un terrain fertile pour la desinformation. **Thumalien** est une solution d'analyse automatisee des posts Bluesky qui permet de :

- **Detecter** les contenus douteux ou trompeurs avec un score de credibilite
- **Analyser** l'impact emotionnel des posts (colere, peur, joie, etc.)
- **Expliquer** pourquoi un contenu est juge suspect (mots influents)
- **Mesurer** l'empreinte energetique de chaque analyse (Green IT)

### Problematique

- Le volume d'information est trop eleve pour une moderation humaine
- Les fake news se propagent plus vite que les dementis
- Les outils de fact-checking existants sont souvent limites a l'anglais
- Peu d'outils prennent en compte l'impact emotionnel

---

## Fonctionnalites

| Fonctionnalite | Description |
|---|---|
| **Collecte de donnees** | API Bluesky (AT Protocol), extraction FR/EN |
| **Pretraitement NLP** | Nettoyage, tokenisation, lemmatisation (spaCy) |
| **Detection fake news** | Classification 3 classes (fiable/douteux/fake) via RoBERTa |
| **Analyse emotionnelle** | 7 emotions (colere, degout, peur, joie, tristesse, surprise, neutre) |
| **Explicabilite IA** | Identification des mots influencant la classification |
| **Dashboard interactif** | Streamlit avec 4 onglets (vue d'ensemble, details, emotions, Green IT) |
| **Suivi energetique** | Monitoring CPU/GPU, emissions CO2, rapport CodeCarbon |

---

## Architecture

```
Bluesky API ──> Collecte ──> Pretraitement NLP ──> Classification ──> Dashboard
                                                  ──> Emotions      ──>
                                                  ──> Explicabilite  ──>
                                                  ──> Energy Tracker ──>
```

Voir [docs/architecture.md](docs/architecture.md) pour le detail complet.

---

## Installation

### Prerequis

- Python 3.10+
- Docker & Docker Compose (optionnel)
- Un compte Bluesky avec un [App Password](https://bsky.app/settings/app-passwords)

### Option 1 : Docker (recommande)

```bash
# Cloner le repo
git clone https://github.com/Luxy133/thumalien.git
cd thumalien

# Configurer les identifiants
cp .env.example .env
# Editer .env avec vos identifiants Bluesky

# Lancer
docker-compose up --build
```

Le dashboard sera accessible sur **http://localhost:8501**

### Option 2 : Installation locale

```bash
# Cloner le repo
git clone https://github.com/Luxy133/thumalien.git
cd thumalien

# Creer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dependances
pip install -r requirements.txt

# Telecharger les modeles spaCy
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm

# Configurer
cp .env.example .env
# Editer .env avec vos identifiants Bluesky
```

---

## Demarrage rapide

### Lancer le dashboard

```bash
streamlit run src/dashboard/app.py
```

### Lancer le pipeline en CLI

```bash
python main.py
```

### Utiliser le notebook d'exploration

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

---

## Utilisation

### Dashboard

1. Ouvrir **http://localhost:8501**
2. Renseigner vos identifiants Bluesky dans la sidebar (ou via `.env`)
3. Entrer un terme de recherche (ex: "elections", "vaccin", "climat")
4. Choisir la langue et le nombre de posts
5. Cliquer sur **Lancer l'analyse**

Le dashboard affiche 4 onglets :

- **Vue d'ensemble** : metriques globales, repartition, top posts suspects
- **Details** : tableau filtrable de tous les posts avec scores
- **Emotions** : distribution, radar chart, croisement emotion/credibilite
- **Green IT** : emissions CO2, consommation energetique par etape

Voir [docs/guide_utilisateur.md](docs/guide_utilisateur.md) pour le guide complet.

### Pipeline Python

```python
from src.pipeline import ThumalienPipeline

pipeline = ThumalienPipeline(
    bluesky_handle="votre-handle.bsky.social",
    bluesky_password="votre-app-password",
)

results = pipeline.run(query="fake news", lang="fr", limit=50)
pipeline.export_results(results)
```

Voir [docs/api.md](docs/api.md) pour la reference API complete.

---

## Stack technique

| Composant | Technologie |
|---|---|
| Collecte | AT Protocol (`atproto`), Python `requests` |
| Pretraitement | spaCy, NLTK |
| Classification | HuggingFace Transformers (RoBERTa) |
| Emotions | DistilRoBERTa (emotion-english-distilroberta-base) |
| Explicabilite | Attention-based (poids des tokens) |
| Stockage | PostgreSQL |
| Dashboard | Streamlit, Plotly |
| Monitoring | CodeCarbon |
| Infrastructure | Docker, Docker Compose |
| Tests | pytest |

---

## Structure du projet

```
thumalien/
├── main.py                          # Point d'entree CLI
├── pyproject.toml                   # Configuration projet
├── requirements.txt                 # Dependances Python
├── Dockerfile                       # Image Docker
├── docker-compose.yml               # Orchestration des services
├── .env.example                     # Template variables d'environnement
│
├── src/
│   ├── pipeline.py                  # Pipeline complet
│   ├── collector/
│   │   └── bluesky_client.py        # Client API Bluesky
│   ├── preprocessing/
│   │   └── text_processor.py        # Nettoyage et tokenisation NLP
│   ├── models/
│   │   ├── fake_news_detector.py    # Classifieur fake news (RoBERTa)
│   │   └── emotion_analyzer.py      # Analyse emotionnelle
│   ├── explainability/
│   │   └── explainer.py             # Explicabilite IA
│   ├── monitoring/
│   │   └── energy_tracker.py        # Suivi energetique
│   └── dashboard/
│       └── app.py                   # Dashboard Streamlit
│
├── tests/                           # Tests unitaires
├── notebooks/                       # Notebooks d'exploration
├── scripts/                         # Scripts SQL et utilitaires
├── data/                            # Donnees (raw, processed, models)
└── docs/                            # Documentation
```

---

## Tests

```bash
# Lancer tous les tests
pytest

# Avec couverture
pytest --cov=src --cov-report=html

# Un module specifique
pytest tests/test_preprocessing.py -v
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Architecture technique, diagrammes, choix techniques |
| [docs/guide_utilisateur.md](docs/guide_utilisateur.md) | Guide utilisateur pas a pas |
| [docs/api.md](docs/api.md) | Reference API et guide developpeur |

---

## KPIs du projet

- **F1-score** de la classification fake news
- **Taux de faux positifs / faux negatifs**
- **Temps moyen d'analyse** par post
- **Emissions CO2** par session d'analyse
- **Consommation energetique** (kWh)

---

## Contribution

1. Fork le projet
2. Creer une branche (`git checkout -b feature/ma-feature`)
3. Commiter (`git commit -m "feat: ajout de ma feature"`)
4. Pousser (`git push origin feature/ma-feature`)
5. Ouvrir une Pull Request

### Conventions

- Commits : [Conventional Commits](https://www.conventionalcommits.org/)
- Code : formatte avec `ruff`
- Tests : obligatoires pour toute nouvelle fonctionnalite

---

## Licence

Projet academique - Master 1 Data & IA
