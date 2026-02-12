"""Modèles Pydantic pour la validation des données du projet Thumalien."""

from datetime import datetime
from pydantic import BaseModel, Field


# ---------- Posts ----------

class PostBase(BaseModel):
    """Données brutes d'un post Bluesky."""
    uri: str
    cid: str
    author_handle: str
    author_display_name: str | None = None
    text: str
    lang: list[str | None] = Field(default_factory=list)
    created_at: str
    collected_at: str
    like_count: int = 0
    repost_count: int = 0
    reply_count: int = 0


class PostProcessed(PostBase):
    """Post après prétraitement NLP."""
    clean_text: str
    tokens: list[str]
    detected_lang: str = "fr"


# ---------- Classification ----------

class CredibilityScores(BaseModel):
    """Scores de crédibilité par catégorie."""
    fiable: float = Field(ge=0, le=1)
    douteux: float = Field(ge=0, le=1)
    fake: float = Field(ge=0, le=1)


class CredibilityResult(BaseModel):
    """Résultat de la classification fake news."""
    label: str = Field(pattern=r"^(fiable|douteux|fake)$")
    confidence: float = Field(ge=0, le=1)
    scores: CredibilityScores


# ---------- Émotions ----------

class EmotionScores(BaseModel):
    """Scores par émotion."""
    colère: float = Field(default=0, ge=0, le=1, alias="colere")
    dégoût: float = Field(default=0, ge=0, le=1, alias="degout")
    peur: float = Field(default=0, ge=0, le=1)
    joie: float = Field(default=0, ge=0, le=1)
    tristesse: float = Field(default=0, ge=0, le=1)
    surprise: float = Field(default=0, ge=0, le=1)
    neutre: float = Field(default=0, ge=0, le=1)

    model_config = {"populate_by_name": True}


class EmotionResult(BaseModel):
    """Résultat de l'analyse émotionnelle."""
    dominant_emotion: str
    confidence: float = Field(ge=0, le=1)
    scores: dict[str, float]


# ---------- Explicabilité ----------

class WordImportance(BaseModel):
    """Importance d'un token pour la classification."""
    token: str
    importance: float
    importance_normalized: float = 0.0


class ExplanationResult(BaseModel):
    """Résultat de l'explicabilité."""
    top_words: list[WordImportance]


# ---------- Analyse complète ----------

class AnalysisResult(BaseModel):
    """Résultat complet de l'analyse d'un post."""
    uri: str
    author_handle: str
    author_display_name: str | None = None
    text: str
    clean_text: str
    tokens: list[str]
    detected_lang: str = "fr"
    like_count: int = 0
    repost_count: int = 0
    reply_count: int = 0
    credibility: CredibilityResult
    emotion: EmotionResult
    explanation: ExplanationResult | None = None


# ---------- Énergie ----------

class EnergyTask(BaseModel):
    """Mesure énergétique d'une tâche."""
    task: str
    start_time: str
    end_time: str
    duration_seconds: float
    emissions_kg_co2: float = 0.0
    energy_kwh: float = 0.0


class EnergySummary(BaseModel):
    """Résumé énergétique complet."""
    total_emissions_kg_co2: float = 0.0
    total_duration_seconds: float = 0.0
    total_energy_kwh: float = 0.0
    num_tasks: int = 0
    tasks: list[EnergyTask] = Field(default_factory=list)


# ---------- Pipeline ----------

class PipelineConfig(BaseModel):
    """Configuration du pipeline."""
    bluesky_handle: str
    bluesky_password: str
    fake_news_model: str = "roberta-base"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    query: str = "fake news"
    lang: str | None = None
    limit: int = Field(default=50, ge=1, le=500)


class PipelineOutput(BaseModel):
    """Sortie complète du pipeline."""
    timestamp: str
    query: str
    num_posts: int
    results: list[AnalysisResult]
    energy: EnergySummary
