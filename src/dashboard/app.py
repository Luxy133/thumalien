"""Dashboard Streamlit pour Thumalien - Détection de Fake News."""

import os
import sys
import logging
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.collector.bluesky_client import BlueskyCollector
from src.preprocessing.text_processor import preprocess_batch
from src.models.fake_news_detector import FakeNewsDetector
from src.models.emotion_analyzer import EmotionAnalyzer
from src.explainability.explainer import PredictionExplainer
from src.monitoring.energy_tracker import EnergyTracker
from src.verification.cross_verifier import CrossSourceVerifier

import random
from datetime import datetime, timezone

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Thumalien - Fake News Detector", page_icon="🔍", layout="wide")

# ---------- Couleurs par label ----------
LABEL_COLORS = {"fiable": "#2ecc71", "douteux": "#f39c12", "fake": "#e74c3c"}
EMOTION_COLORS = {
    "colère": "#e74c3c", "dégoût": "#8e44ad", "peur": "#2c3e50",
    "joie": "#f1c40f", "tristesse": "#3498db", "surprise": "#e67e22", "neutre": "#95a5a6",
}


# ---------- Données démo ----------
DEMO_POSTS = [
    {"text": "Le ministre de la Santé a confirmé l'extension du programme de vaccination pour les 12-18 ans, conformément aux recommandations de la HAS.", "author_handle": "info-sante.bsky.social", "label": "fiable", "emotion": "neutre", "lang_tag": "fr"},
    {"text": "URGENT !! Le gouvernement cache la vérité sur les effets secondaires des vaccins ! Des milliers de morts dissimulés ! Partagez avant censure !!!", "author_handle": "verite-cache.bsky.social", "label": "fake", "emotion": "peur", "lang_tag": "fr"},
    {"text": "Selon une étude de l'INSERM publiée dans The Lancet, le nouveau traitement réduit la mortalité de 30% chez les patients à risque.", "author_handle": "sciences-actu.bsky.social", "label": "fiable", "emotion": "joie", "lang_tag": "fr"},
    {"text": "On nous ment depuis le début ! Les élites mondiales contrôlent tout avec la 5G. RÉVEILLEZ-VOUS !", "author_handle": "reveil-citoyen.bsky.social", "label": "fake", "emotion": "colère", "lang_tag": "fr"},
    {"text": "Le taux de chômage a diminué de 0.3% ce trimestre selon les chiffres de l'INSEE.", "author_handle": "eco-france.bsky.social", "label": "fiable", "emotion": "neutre", "lang_tag": "fr"},
    {"text": "Des sources proches du dossier évoquent une possible réforme, mais aucune confirmation officielle n'a été donnée.", "author_handle": "media-info.bsky.social", "label": "douteux", "emotion": "surprise", "lang_tag": "fr"},
    {"text": "BREAKING: Un scientifique renommé affirme que boire du jus de citron guérit le cancer. Big Pharma ne veut pas que vous sachiez !", "author_handle": "sante-naturelle.bsky.social", "label": "fake", "emotion": "surprise", "lang_tag": "fr"},
    {"text": "L'Assemblée nationale a voté la loi sur la transition énergétique avec 342 voix pour et 128 contre.", "author_handle": "politique-fr.bsky.social", "label": "fiable", "emotion": "neutre", "lang_tag": "fr"},
    {"text": "Certains experts remettent en question les chiffres officiels du PIB, parlant de méthodologie contestable.", "author_handle": "debat-eco.bsky.social", "label": "douteux", "emotion": "tristesse", "lang_tag": "fr"},
    {"text": "La mairie a inauguré le nouveau parc urbain de 5 hectares dans le quartier nord. 200 arbres ont été plantés.", "author_handle": "ville-info.bsky.social", "label": "fiable", "emotion": "joie", "lang_tag": "fr"},
    {"text": "Les chemtrails sont la preuve que le gouvernement empoisonne la population ! Regardez le ciel et ouvrez les yeux !!!", "author_handle": "complot-alert.bsky.social", "label": "fake", "emotion": "colère", "lang_tag": "fr"},
    {"text": "Le rapport du GIEC indique une hausse des températures de 1.5°C d'ici 2030 si les émissions ne sont pas réduites.", "author_handle": "climat-actu.bsky.social", "label": "fiable", "emotion": "peur", "lang_tag": "fr"},
    {"text": "Une vidéo virale prétend montrer un OVNI à Toulouse. Les experts n'ont pas encore confirmé l'authenticité des images.", "author_handle": "buzz-france.bsky.social", "label": "douteux", "emotion": "surprise", "lang_tag": "fr"},
    {"text": "Le match PSG-Marseille s'est terminé sur un score de 2-1, avec un doublé de la recrue star.", "author_handle": "sports-live.bsky.social", "label": "fiable", "emotion": "joie", "lang_tag": "fr"},
    {"text": "EXCLUSIF : Un ancien employé révèle que les résultats des élections ont été truqués par un logiciel secret !!!", "author_handle": "scoop-politique.bsky.social", "label": "fake", "emotion": "colère", "lang_tag": "fr"},
    {"text": "La BCE maintient ses taux d'intérêt inchangés, conformément aux attentes des analystes.", "author_handle": "finance-eu.bsky.social", "label": "fiable", "emotion": "neutre", "lang_tag": "fr"},
    {"text": "Des témoins affirment avoir vu des phénomènes étranges près de la centrale. L'exploitant dément tout incident.", "author_handle": "local-news.bsky.social", "label": "douteux", "emotion": "peur", "lang_tag": "fr"},
    {"text": "SCANDALE ! Les politiciens se sont augmentés de 50% en secret pendant que les Français souffrent !", "author_handle": "indignes.bsky.social", "label": "fake", "emotion": "colère", "lang_tag": "fr"},
    {"text": "Le festival de Cannes a décerné la Palme d'Or au réalisateur japonais pour son film sur l'immigration.", "author_handle": "culture-actu.bsky.social", "label": "fiable", "emotion": "joie", "lang_tag": "fr"},
    {"text": "Un article circule affirmant que le wifi cause des tumeurs cérébrales. L'OMS n'a pas classé le wifi comme cancérigène.", "author_handle": "fact-check-fr.bsky.social", "label": "douteux", "emotion": "neutre", "lang_tag": "fr"},
    {"text": "Magnifique victoire de l'équipe de France en finale ! Un moment historique pour le sport français !", "author_handle": "sport-passion.bsky.social", "label": "fiable", "emotion": "joie", "lang_tag": "fr"},
    {"text": "ILS NE VEULENT PAS QUE VOUS VOYIEZ ÇA : les preuves irréfutables que la Terre est plate sont enfin révélées !", "author_handle": "terre-plate-fr.bsky.social", "label": "fake", "emotion": "surprise", "lang_tag": "fr"},
    {"text": "Le prix du pétrole a chuté de 5% suite aux annonces de l'OPEP sur l'augmentation de la production.", "author_handle": "marches-info.bsky.social", "label": "fiable", "emotion": "neutre", "lang_tag": "fr"},
    {"text": "Cette situation est vraiment déprimante. Les hôpitaux sont surchargés et personne ne fait rien.", "author_handle": "citoyen-lambda.bsky.social", "label": "douteux", "emotion": "tristesse", "lang_tag": "fr"},
    {"text": "Attention arnaque ! Un faux site gouvernemental collecte vos données personnelles. Ne cliquez pas sur ce lien.", "author_handle": "cyber-alerte.bsky.social", "label": "fiable", "emotion": "peur", "lang_tag": "fr"},
    # --- Posts anglais ---
    {"text": "The WHO confirmed the new vaccine is safe and effective for all age groups.", "author_handle": "health-news.bsky.social", "label": "fiable", "emotion": "neutre", "lang_tag": "en"},
    {"text": "SHOCKING!! Government is hiding alien technology in Area 51!!! Share before they delete this!!!", "author_handle": "truth-seeker.bsky.social", "label": "fake", "emotion": "peur", "lang_tag": "en"},
    {"text": "According to a Stanford study published in Nature, the new treatment reduces mortality by 25%.", "author_handle": "science-daily.bsky.social", "label": "fiable", "emotion": "joie", "lang_tag": "en"},
    {"text": "WAKE UP PEOPLE! The moon landing was FAKED by NASA. Here's the proof they don't want you to see!!!", "author_handle": "conspiracy-hub.bsky.social", "label": "fake", "emotion": "colère", "lang_tag": "en"},
    {"text": "The Federal Reserve kept interest rates unchanged, in line with market expectations.", "author_handle": "finance-wire.bsky.social", "label": "fiable", "emotion": "neutre", "lang_tag": "en"},
    {"text": "Some researchers question the methodology behind the latest GDP figures, calling them unreliable.", "author_handle": "econ-debate.bsky.social", "label": "douteux", "emotion": "tristesse", "lang_tag": "en"},
    {"text": "BREAKING: Secret documents reveal the election was rigged by a shadow organization!!!", "author_handle": "deep-state-watch.bsky.social", "label": "fake", "emotion": "colère", "lang_tag": "en"},
    {"text": "NASA's James Webb telescope captured stunning new images of a distant galaxy cluster.", "author_handle": "space-news.bsky.social", "label": "fiable", "emotion": "surprise", "lang_tag": "en"},
    {"text": "A viral video claims to show a UFO over London. Experts have not yet verified the footage.", "author_handle": "uk-buzz.bsky.social", "label": "douteux", "emotion": "surprise", "lang_tag": "en"},
    {"text": "The Premier League final ended 3-2 in a dramatic last-minute goal by the home team.", "author_handle": "sports-uk.bsky.social", "label": "fiable", "emotion": "joie", "lang_tag": "en"},
]


def generate_demo_results():
    """Génère des résultats d'analyse réalistes à partir des données démo."""
    random.seed(42)
    results = []
    emotions_list = ["colère", "dégoût", "peur", "joie", "tristesse", "surprise", "neutre"]

    for i, post in enumerate(DEMO_POSTS):
        # Scores de crédibilité réalistes selon le label
        if post["label"] == "fiable":
            scores = {"fiable": round(random.uniform(0.65, 0.95), 3), "douteux": 0, "fake": 0}
            scores["douteux"] = round(random.uniform(0.02, 0.20), 3)
            scores["fake"] = round(1 - scores["fiable"] - scores["douteux"], 3)
        elif post["label"] == "douteux":
            scores = {"douteux": round(random.uniform(0.45, 0.75), 3), "fiable": 0, "fake": 0}
            scores["fiable"] = round(random.uniform(0.10, 0.30), 3)
            scores["fake"] = round(1 - scores["douteux"] - scores["fiable"], 3)
        else:
            scores = {"fake": round(random.uniform(0.60, 0.95), 3), "douteux": 0, "fiable": 0}
            scores["douteux"] = round(random.uniform(0.02, 0.25), 3)
            scores["fiable"] = round(max(0, 1 - scores["fake"] - scores["douteux"]), 3)

        # Scores émotionnels réalistes
        dominant = post["emotion"]
        emo_scores = {}
        remaining = 1.0
        dominant_score = round(random.uniform(0.45, 0.85), 3)
        emo_scores[dominant] = dominant_score
        remaining -= dominant_score
        other_emos = [e for e in emotions_list if e != dominant]
        for j, e in enumerate(other_emos):
            if j == len(other_emos) - 1:
                emo_scores[e] = round(max(0, remaining), 3)
            else:
                s = round(random.uniform(0, remaining / 2), 3)
                emo_scores[e] = s
                remaining -= s

        # Mots influents pour les posts douteux/fake
        explanation = None
        if post["label"] in ("douteux", "fake"):
            words = post["text"].lower().split()
            suspicious_words = [w.strip("!.,?:;\"'()") for w in words if len(w) > 4][:8]
            random.shuffle(suspicious_words)
            explanation = {
                "top_words": [
                    {"token": w, "importance": round(random.uniform(0.3, 1.0), 3),
                     "importance_normalized": round(random.uniform(0.4, 1.0), 3)}
                    for w in suspicious_words[:5]
                ]
            }

        # Vérification croisée synthétique
        if post["label"] == "fiable":
            v_score = round(random.uniform(0.6, 0.95), 3)
        elif post["label"] == "fake":
            v_score = round(random.uniform(0.05, 0.35), 3)
        else:
            v_score = round(random.uniform(0.3, 0.65), 3)

        v_confidence = round(random.uniform(0.3, 0.9), 3)
        n_sources = random.randint(1, 4)

        verification = {
            "sources": {
                "web_search": {
                    "source_name": "web_search", "available": True,
                    "num_results": random.randint(1, 5),
                    "corroboration_score": round(random.uniform(0.2, 0.9), 3),
                    "confidence": round(random.uniform(0.3, 0.8), 3),
                    "top_sources": [], "error": None,
                },
                "fact_check": {
                    "source_name": "fact_check", "available": random.choice([True, False]),
                    "num_results": random.randint(0, 3),
                    "corroboration_score": round(v_score + random.uniform(-0.1, 0.1), 3),
                    "confidence": round(random.uniform(0.2, 0.9), 3),
                    "top_sources": [], "error": None,
                },
                "news": {
                    "source_name": "news", "available": True,
                    "num_results": random.randint(1, 5),
                    "corroboration_score": round(v_score + random.uniform(-0.15, 0.15), 3),
                    "confidence": round(random.uniform(0.3, 0.85), 3),
                    "top_sources": [], "error": None,
                },
                "bluesky": {
                    "source_name": "bluesky", "available": random.choice([True, False]),
                    "num_results": random.randint(0, 10),
                    "corroboration_score": round(random.uniform(0.2, 0.8), 3),
                    "confidence": round(random.uniform(0.1, 0.6), 3),
                    "top_sources": [], "error": None,
                },
            },
            "verification_score": v_score,
            "verification_confidence": v_confidence,
            "num_sources_available": n_sources,
        }

        results.append({
            "uri": f"at://did:plc:demo{i}/app.bsky.feed.post/{i:03d}",
            "cid": f"bafyreidemo{i:04d}",
            "author_handle": post["author_handle"],
            "author_display_name": post["author_handle"].split(".")[0].replace("-", " ").title(),
            "text": post["text"],
            "clean_text": post["text"].lower(),
            "tokens": post["text"].lower().split()[:10],
            "detected_lang": post["lang_tag"],
            "lang": [post["lang_tag"]],
            "created_at": "2025-06-15T10:00:00Z",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "like_count": random.randint(0, 500),
            "repost_count": random.randint(0, 200),
            "reply_count": random.randint(0, 80),
            "credibility": {
                "label": post["label"],
                "confidence": scores[post["label"]],
                "scores": scores,
            },
            "emotion": {
                "dominant_emotion": dominant,
                "confidence": emo_scores[dominant],
                "scores": emo_scores,
            },
            "explanation": explanation,
            "verification": verification,
        })

    energy = {
        "total_emissions_kg_co2": 0.000342,
        "total_duration_seconds": 12.7,
        "total_energy_kwh": 0.000891,
        "num_tasks": 5,
        "tasks": [
            {"task": "collecte", "duration_seconds": 2.1, "emissions_kg_co2": 0.000021, "energy_kwh": 0.000054},
            {"task": "pretraitement", "duration_seconds": 1.3, "emissions_kg_co2": 0.000015, "energy_kwh": 0.000039},
            {"task": "classification", "duration_seconds": 4.8, "emissions_kg_co2": 0.000156, "energy_kwh": 0.000405},
            {"task": "emotion", "duration_seconds": 3.2, "emissions_kg_co2": 0.000112, "energy_kwh": 0.000291},
            {"task": "explicabilite", "duration_seconds": 1.3, "emissions_kg_co2": 0.000038, "energy_kwh": 0.000102},
        ],
    }

    return results, energy


# ---------- Chargement des modèles (mis en cache) ----------
@st.cache_resource(show_spinner="Chargement du détecteur de fake news...")
def load_detector():
    return FakeNewsDetector()


@st.cache_resource(show_spinner="Chargement de l'analyseur d'émotions...")
def load_emotion_analyzer():
    return EmotionAnalyzer()


@st.cache_resource(show_spinner="Connexion à Bluesky...")
def load_collector(handle: str, password: str):
    return BlueskyCollector(handle, password)


# ---------- Pipeline d'analyse ----------
def run_analysis(collector, detector, emotion_analyzer, query, lang, limit):
    """Exécute le pipeline complet et retourne résultats + rapport énergie."""
    tracker = EnergyTracker()

    # 1. Collecte
    with tracker.track("collecte"):
        raw_posts = collector.search_posts(query, lang=lang, limit=limit)

    if not raw_posts:
        return [], tracker.get_summary()

    # 2. Prétraitement
    with tracker.track("pretraitement"):
        processed = preprocess_batch(raw_posts)

    texts = [p["clean_text"] for p in processed]

    # 3. Classification
    with tracker.track("classification"):
        predictions = detector.predict_batch(texts)

    # 4. Vérification croisée
    cross_verifier = CrossSourceVerifier(collector=collector)
    with tracker.track("verification"):
        verifications = cross_verifier.verify_batch(texts)

    # 5. Émotions
    with tracker.track("emotion"):
        emotions = emotion_analyzer.analyze_batch(texts)

    # 6. Explicabilité
    explainer = PredictionExplainer(detector.model, detector.tokenizer)
    with tracker.track("explicabilite"):
        results = []
        for post, pred, emo, verif in zip(processed, predictions, emotions, verifications):
            # Combiner scores si vérification disponible
            credibility = pred
            if verif and verif.get("verification_confidence", 0) > 0:
                combined_scores = CrossSourceVerifier.combine_with_credibility(
                    pred["scores"],
                    verif["verification_score"],
                    verif["verification_confidence"],
                )
                combined_label = max(combined_scores, key=combined_scores.get)
                credibility = {
                    "label": combined_label,
                    "confidence": pred["confidence"],
                    "scores": combined_scores,
                    "original_scores": pred["scores"],
                }

            entry = {**post, "credibility": credibility, "emotion": emo, "explanation": None, "verification": verif}
            if credibility["label"] in ("douteux", "fake") and credibility["confidence"] > 0.6:
                expl = explainer.explain(post["clean_text"])
                entry["explanation"] = {"top_words": expl["top_influential_words"][:5]}
            results.append(entry)

    tracker.save_report()
    return results, tracker.get_summary()


# ---------- Fonctions d'affichage ----------
def render_overview(results):
    """Onglet Vue d'ensemble."""
    st.header("Score de crédibilité global")

    total = len(results)
    fiable = sum(1 for r in results if r["credibility"]["label"] == "fiable")
    douteux = sum(1 for r in results if r["credibility"]["label"] == "douteux")
    fake = sum(1 for r in results if r["credibility"]["label"] == "fake")

    # Confiance moyenne globale et par label
    avg_confidence = sum(r["credibility"]["confidence"] for r in results) / total
    avg_conf_fiable = (
        sum(r["credibility"]["confidence"] for r in results if r["credibility"]["label"] == "fiable") / fiable
        if fiable else 0
    )
    avg_conf_douteux = (
        sum(r["credibility"]["confidence"] for r in results if r["credibility"]["label"] == "douteux") / douteux
        if douteux else 0
    )
    avg_conf_fake = (
        sum(r["credibility"]["confidence"] for r in results if r["credibility"]["label"] == "fake") / fake
        if fake else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Posts analysés", total)
    col2.metric("Fiables", fiable, delta=f"{fiable / total * 100:.0f}%")
    col3.metric("Douteux", douteux, delta=f"{douteux / total * 100:.0f}%", delta_color="off")
    col4.metric("Fake News", fake, delta=f"{fake / total * 100:.0f}%", delta_color="inverse")
    col5.metric("Confiance moyenne", f"{avg_confidence:.0%}")

    # Confiance moyenne par label
    st.subheader("Taux de confiance par catégorie")
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    conf_col1.metric("Confiance Fiable", f"{avg_conf_fiable:.1%}", help="Confiance moyenne du modèle pour les posts classés fiables")
    conf_col2.metric("Confiance Douteux", f"{avg_conf_douteux:.1%}", help="Confiance moyenne du modèle pour les posts classés douteux")
    conf_col3.metric("Confiance Fake", f"{avg_conf_fake:.1%}", help="Confiance moyenne du modèle pour les posts classés fake")

    # Jauge de confiance globale
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_confidence * 100,
        title={"text": "Confiance globale du modèle"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3498db"},
            "steps": [
                {"range": [0, 40], "color": "#fadbd8"},
                {"range": [40, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#d5f5e3"},
            ],
            "threshold": {"line": {"color": "black", "width": 2}, "value": avg_confidence * 100},
        },
    ))
    fig_gauge.update_layout(height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Graphique de répartition
    col_pie, col_conf = st.columns(2)

    with col_pie:
        labels_count = pd.DataFrame({"Label": ["fiable", "douteux", "fake"], "Nombre": [fiable, douteux, fake]})
        fig = px.pie(
            labels_count, names="Label", values="Nombre",
            color="Label", color_discrete_map=LABEL_COLORS,
            title="Répartition de la crédibilité",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_conf:
        confidences = [r["credibility"]["confidence"] for r in results]
        labels = [r["credibility"]["label"] for r in results]
        fig = px.histogram(
            x=confidences, color=labels, color_discrete_map=LABEL_COLORS,
            nbins=20, title="Distribution des scores de confiance",
            labels={"x": "Confiance", "color": "Label"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top posts les plus douteux
    st.subheader("Posts les plus suspects")
    suspicious = sorted(results, key=lambda r: r["credibility"]["scores"].get("fake", 0), reverse=True)[:5]
    for i, post in enumerate(suspicious, 1):
        score = post["credibility"]["scores"]["fake"]
        confidence = post["credibility"]["confidence"]
        label = post["credibility"]["label"]
        color = LABEL_COLORS[label]
        with st.expander(f"#{i} — @{post['author_handle']} — :{color}[{label.upper()}] ({score:.0%})"):
            st.markdown(f"**Texte original :** {post['text']}")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Confiance", f"{confidence:.1%}")
            mc2.metric("Émotion", post["emotion"]["dominant_emotion"])
            mc3.metric("Engagement", f"{post.get('like_count', 0)} likes")
            # Barre de confiance par catégorie
            st.markdown("**Scores de crédibilité :**")
            for lbl in ["fiable", "douteux", "fake"]:
                s = post["credibility"]["scores"].get(lbl, 0)
                st.progress(s, text=f"{lbl.capitalize()} : {s:.1%}")
            if post.get("explanation"):
                words = ", ".join(w["token"] for w in post["explanation"]["top_words"])
                st.markdown(f"**Mots influents :** `{words}`")


def render_details(results):
    """Onglet Détails par post."""
    st.header("Détails par post")

    rows = []
    for r in results:
        top_words = ""
        if r.get("explanation") and r["explanation"].get("top_words"):
            top_words = ", ".join(w["token"] for w in r["explanation"]["top_words"])

        rows.append({
            "Auteur": f"@{r['author_handle']}",
            "Texte": r["text"][:120] + ("..." if len(r["text"]) > 120 else ""),
            "Label": r["credibility"]["label"],
            "Confiance": r["credibility"]["confidence"],
            "Score Fiable": r["credibility"]["scores"].get("fiable", 0),
            "Score Fake": r["credibility"]["scores"].get("fake", 0),
            "Émotion": r["emotion"]["dominant_emotion"],
            "Conf. Émotion": r["emotion"]["confidence"],
            "Mots influents": top_words,
            "Likes": r.get("like_count", 0),
            "Reposts": r.get("repost_count", 0),
        })

    df = pd.DataFrame(rows)

    # Filtre par label
    selected_labels = st.multiselect(
        "Filtrer par crédibilité", ["fiable", "douteux", "fake"],
        default=["fiable", "douteux", "fake"],
    )
    df_filtered = df[df["Label"].isin(selected_labels)]

    # Filtre par confiance
    min_conf = st.slider("Confiance minimum", 0.0, 1.0, 0.0, 0.05, format="%.0f%%",
                         help="Filtrer les posts avec un score de confiance minimum")
    df_filtered = df_filtered[df_filtered["Confiance"] >= min_conf]

    st.dataframe(
        df_filtered,
        use_container_width=True,
        column_config={
            "Label": st.column_config.TextColumn("Label", width="small"),
            "Confiance": st.column_config.ProgressColumn("Confiance", min_value=0, max_value=1, format="%.0f%%"),
            "Score Fiable": st.column_config.ProgressColumn("Score Fiable", min_value=0, max_value=1, format="%.0f%%"),
            "Score Fake": st.column_config.ProgressColumn("Score Fake", min_value=0, max_value=1, format="%.0f%%"),
            "Conf. Émotion": st.column_config.ProgressColumn("Conf. Émotion", min_value=0, max_value=1, format="%.0f%%"),
            "Émotion": st.column_config.TextColumn("Émotion", width="small"),
        },
    )
    st.caption(f"{len(df_filtered)} posts affichés sur {len(df)} au total")


def render_emotions(results):
    """Onglet Analyse émotionnelle."""
    st.header("Analyse émotionnelle")

    # Distribution globale des émotions dominantes
    emotion_counts = {}
    for r in results:
        emo = r["emotion"]["dominant_emotion"]
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    col_bar, col_radar = st.columns(2)

    with col_bar:
        emo_df = pd.DataFrame(list(emotion_counts.items()), columns=["Émotion", "Nombre"])
        emo_df = emo_df.sort_values("Nombre", ascending=False)
        fig = px.bar(
            emo_df, x="Émotion", y="Nombre", color="Émotion",
            color_discrete_map=EMOTION_COLORS,
            title="Émotions dominantes dans les posts",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_radar:
        # Scores moyens par émotion
        all_emotions = {}
        for r in results:
            for emo, score in r["emotion"]["scores"].items():
                all_emotions.setdefault(emo, []).append(score)
        avg_scores = {emo: sum(vals) / len(vals) for emo, vals in all_emotions.items()}

        fig = go.Figure(data=go.Scatterpolar(
            r=list(avg_scores.values()),
            theta=list(avg_scores.keys()),
            fill="toself",
            name="Score moyen",
        ))
        fig.update_layout(title="Profil émotionnel moyen", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)

    # Croisement émotion x crédibilité
    st.subheader("Émotions par catégorie de crédibilité")
    cross_data = []
    for r in results:
        cross_data.append({
            "Label": r["credibility"]["label"],
            "Émotion dominante": r["emotion"]["dominant_emotion"],
        })
    cross_df = pd.DataFrame(cross_data)
    fig = px.histogram(
        cross_df, x="Émotion dominante", color="Label",
        color_discrete_map=LABEL_COLORS, barmode="group",
        title="Quelles émotions sont associées aux fake news ?",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_energy(energy_summary):
    """Onglet Green IT."""
    st.header("Suivi énergétique (Green IT)")

    total_co2 = energy_summary.get("total_emissions_kg_co2", 0)
    total_kwh = energy_summary.get("total_energy_kwh", 0)
    total_duration = energy_summary.get("total_duration_seconds", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Émissions CO2", f"{total_co2:.6f} kg")
    col2.metric("Énergie consommée", f"{total_kwh:.6f} kWh")
    col3.metric("Durée totale", f"{total_duration:.1f} s")

    tasks = energy_summary.get("tasks", [])
    if tasks:
        # Répartition par tâche
        col_bar, col_pie = st.columns(2)

        task_df = pd.DataFrame(tasks)

        with col_bar:
            fig = px.bar(
                task_df, x="task", y="duration_seconds", color="task",
                title="Durée par étape du pipeline",
                labels={"task": "Étape", "duration_seconds": "Durée (s)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_pie:
            fig = px.pie(
                task_df, names="task", values="emissions_kg_co2",
                title="Répartition des émissions CO2 par étape",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Détail par tâche")
        st.dataframe(task_df[["task", "duration_seconds", "emissions_kg_co2", "energy_kwh"]], use_container_width=True)


def render_history():
    """Onglet Historique des analyses."""
    st.header("Historique des analyses")

    try:
        from src.database.connection import get_session
        from src.database.repository import SessionRepository

        db_session = get_session()
        repo = SessionRepository(db_session)
        sessions = repo.get_history(limit=20)
        db_session.close()

        if not sessions:
            st.info("Aucune analyse sauvegardée. Coche 'Sauvegarder en BDD' dans la sidebar.")
            return

        rows = []
        for s in sessions:
            rows.append({
                "Date": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "?",
                "Requête": s.query,
                "Langue": s.lang or "Toutes",
                "Posts": s.num_posts,
                "Fiables": s.num_fiable,
                "Douteux": s.num_douteux,
                "Fake": s.num_fake,
                "CO2 (kg)": f"{s.total_emissions_co2:.6f}",
                "Durée (s)": f"{s.duration_seconds:.1f}",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Graphique d'évolution
        if len(sessions) > 1:
            hist_df = pd.DataFrame(rows)
            fig = px.bar(
                hist_df, x="Date", y=["Fiables", "Douteux", "Fake"],
                barmode="stack", title="Évolution des analyses",
                color_discrete_map={"Fiables": "#2ecc71", "Douteux": "#f39c12", "Fake": "#e74c3c"},
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Base de données non disponible : {e}")
        st.info("Lance PostgreSQL avec `docker-compose up db` et initialise avec `make db-init`.")


def render_verification(results):
    """Onglet Vérification croisée des sources."""
    st.header("Vérification croisée des sources")

    verified = [r for r in results if r.get("verification")]
    if not verified:
        st.info("Aucune donnée de vérification disponible. Lancez une analyse avec la vérification activée.")
        return

    # Jauge de corroboration globale
    avg_v_score = sum(r["verification"]["verification_score"] for r in verified) / len(verified)
    avg_v_conf = sum(r["verification"]["verification_confidence"] for r in verified) / len(verified)
    avg_sources = sum(r["verification"]["num_sources_available"] for r in verified) / len(verified)

    col1, col2, col3 = st.columns(3)
    col1.metric("Score corroboration moyen", f"{avg_v_score:.1%}")
    col2.metric("Confiance vérification", f"{avg_v_conf:.1%}")
    col3.metric("Sources disponibles (moy.)", f"{avg_sources:.1f}")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_v_score * 100,
        title={"text": "Corroboration globale"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#9b59b6"},
            "steps": [
                {"range": [0, 30], "color": "#fadbd8"},
                {"range": [30, 60], "color": "#fdebd0"},
                {"range": [60, 100], "color": "#d5f5e3"},
            ],
            "threshold": {"line": {"color": "black", "width": 2}, "value": avg_v_score * 100},
        },
    ))
    fig_gauge.update_layout(height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Bar chart par source
    st.subheader("Scores par source de vérification")
    source_names = ["web_search", "fact_check", "news", "bluesky"]
    source_labels = {"web_search": "Web Search", "fact_check": "Fact Check", "news": "Actualités", "bluesky": "Bluesky"}
    source_colors = {"web_search": "#3498db", "fact_check": "#e74c3c", "news": "#2ecc71", "bluesky": "#9b59b6"}

    avg_by_source = {}
    for sname in source_names:
        scores = []
        for r in verified:
            src_data = r["verification"].get("sources", {}).get(sname, {})
            if src_data.get("available"):
                scores.append(src_data.get("corroboration_score", 0.5))
        avg_by_source[sname] = sum(scores) / len(scores) if scores else 0

    source_df = pd.DataFrame([
        {"Source": source_labels.get(k, k), "Score": v, "source_key": k}
        for k, v in avg_by_source.items()
    ])
    fig = px.bar(
        source_df, x="Source", y="Score", color="Source",
        color_discrete_map={source_labels[k]: v for k, v in source_colors.items()},
        title="Score de corroboration moyen par source",
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: Score modèle vs Score vérification
    st.subheader("Modèle vs Vérification")
    scatter_data = []
    for r in verified:
        scatter_data.append({
            "Score Fiable (modèle)": r["credibility"]["scores"].get("fiable", 0),
            "Score Vérification": r["verification"]["verification_score"],
            "Label": r["credibility"]["label"],
            "Auteur": r.get("author_handle", ""),
        })
    scatter_df = pd.DataFrame(scatter_data)
    fig = px.scatter(
        scatter_df, x="Score Fiable (modèle)", y="Score Vérification",
        color="Label", color_discrete_map=LABEL_COLORS,
        hover_data=["Auteur"],
        title="Corrélation entre score du modèle et vérification croisée",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Détail par post
    st.subheader("Détail par post")
    for i, r in enumerate(verified[:10], 1):
        v = r["verification"]
        label = r["credibility"]["label"]
        color = LABEL_COLORS.get(label, "#95a5a6")
        with st.expander(f"#{i} — @{r.get('author_handle', '?')} — :{color}[{label.upper()}] — Vérif: {v['verification_score']:.0%}"):
            st.markdown(f"**Texte :** {r['text'][:200]}")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Score vérification", f"{v['verification_score']:.1%}")
            mc2.metric("Confiance", f"{v['verification_confidence']:.1%}")
            mc3.metric("Sources", v["num_sources_available"])

            for sname in source_names:
                src = v.get("sources", {}).get(sname, {})
                if src.get("available"):
                    st.progress(
                        src.get("corroboration_score", 0.5),
                        text=f"{source_labels.get(sname, sname)} : {src.get('corroboration_score', 0.5):.1%} (confiance: {src.get('confidence', 0):.1%})",
                    )


# ---------- App principale ----------
def main():
    st.title("Thumalien - Détection de Fake News sur Bluesky")
    st.markdown("Analyse automatisée des posts Bluesky : crédibilité, émotions et impact.")

    # Sidebar — Configuration
    with st.sidebar:
        st.header("Configuration")

        handle = os.getenv("BLUESKY_HANDLE", "")
        password = os.getenv("BLUESKY_PASSWORD", "")

        with st.expander("Identifiants Bluesky", expanded=not bool(handle)):
            handle = st.text_input("Handle", value=handle, placeholder="votre-handle.bsky.social")
            password = st.text_input("App Password", value=password, type="password")

        st.divider()

        # Mode démo
        demo_btn = st.button("Mode Démo (sans compte)", type="secondary", use_container_width=True)

        st.divider()
        search_query = st.text_input("Recherche", placeholder="Ex: élections, santé, vaccin...")
        lang_options = {"Toutes": None, "Français": "fr", "English": "en"}
        lang_filter = st.selectbox("Langue", list(lang_options.keys()))
        num_posts = st.slider("Nombre de posts", 10, 200, 50)
        save_to_db = st.checkbox("Sauvegarder en BDD", value=False)
        analyze_btn = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    # Mode démo
    if demo_btn:
        with st.status("Chargement de la démo...", expanded=True) as status:
            st.write(f"Génération de {len(DEMO_POSTS)} posts d'exemple...")
            results, energy = generate_demo_results()
            st.session_state["results"] = results
            st.session_state["energy"] = energy
            status.update(label=f"Démo chargée — {len(results)} posts", state="complete")

    # Lancement de l'analyse réelle
    elif analyze_btn:
        if not handle or not password:
            st.error("Renseigne tes identifiants Bluesky dans la sidebar, ou utilise le **Mode Démo**.")
            return
        if not search_query:
            st.error("Entre un terme de recherche.")
            return

        with st.status("Analyse en cours...", expanded=True) as status:
            st.write("Connexion à Bluesky...")
            try:
                collector = load_collector(handle, password)
            except Exception as e:
                st.error(f"Erreur de connexion à Bluesky : {e}")
                return

            st.write("Chargement des modèles IA...")
            detector = load_detector()
            emotion_analyzer = load_emotion_analyzer()

            st.write(f"Collecte des posts pour « {search_query} »...")
            lang = lang_options[lang_filter]

            try:
                results, energy = run_analysis(collector, detector, emotion_analyzer, search_query, lang, num_posts)
            except Exception as e:
                st.error(f"Erreur pendant l'analyse : {e}")
                logger.exception("Erreur pipeline")
                return

            st.session_state["results"] = results
            st.session_state["energy"] = energy
            st.session_state["last_query"] = search_query
            st.session_state["last_lang"] = lang

            # Sauvegarde en BDD si demandé
            if save_to_db and results:
                try:
                    from src.database.connection import get_session
                    from src.database.repository import save_pipeline_results
                    db_session = get_session()
                    save_pipeline_results(db_session, search_query, lang, results, energy)
                    db_session.close()
                    st.write("Résultats sauvegardés en base de données.")
                except Exception as db_err:
                    st.warning(f"Sauvegarde BDD échouée : {db_err}")

            status.update(label=f"Analyse terminée — {len(results)} posts", state="complete")

    # Affichage des résultats
    results = st.session_state.get("results")
    energy = st.session_state.get("energy")

    if not results:
        tab_overview, tab_details, tab_emotions, tab_verification, tab_energy, tab_history = st.tabs(
            ["Vue d'ensemble", "Détails", "Émotions", "Vérification", "Green IT", "Historique"]
        )
        with tab_overview:
            st.info("Lance une analyse depuis la sidebar pour voir les résultats.")
        with tab_history:
            render_history()
        return

    # --- Filtre par langue ---
    n_fr = sum(1 for r in results if r.get("detected_lang") == "fr")
    n_en = sum(1 for r in results if r.get("detected_lang") == "en")
    lang_view = st.radio(
        "Vue par langue",
        ["Toutes les langues", "Français", "English"],
        horizontal=True,
    )
    lang_filter_map = {"Toutes les langues": None, "Français": "fr", "English": "en"}
    selected_lang = lang_filter_map[lang_view]

    if selected_lang:
        filtered_results = [r for r in results if r.get("detected_lang") == selected_lang]
    else:
        filtered_results = results

    st.caption(
        f"{len(filtered_results)} posts affichés — "
        f"Français : {n_fr} | English : {n_en} | Total : {len(results)}"
    )

    tab_overview, tab_details, tab_emotions, tab_verification, tab_energy, tab_history = st.tabs(
        ["Vue d'ensemble", "Détails", "Émotions", "Vérification", "Green IT", "Historique"]
    )

    with tab_overview:
        render_overview(filtered_results)

    with tab_details:
        render_details(filtered_results)

    with tab_emotions:
        render_emotions(filtered_results)

    with tab_verification:
        render_verification(filtered_results)

    with tab_energy:
        render_energy(energy)

    with tab_history:
        render_history()


if __name__ == "__main__":
    main()
