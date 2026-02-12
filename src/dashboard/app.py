"""Dashboard Streamlit pour Thumalien - Détection de Fake News."""

import os
import logging

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

    # 4. Émotions
    with tracker.track("emotion"):
        emotions = emotion_analyzer.analyze_batch(texts)

    # 5. Explicabilité
    explainer = PredictionExplainer(detector.model, detector.tokenizer)
    with tracker.track("explicabilite"):
        results = []
        for post, pred, emo in zip(processed, predictions, emotions):
            entry = {**post, "credibility": pred, "emotion": emo, "explanation": None}
            if pred["label"] in ("douteux", "fake") and pred["confidence"] > 0.6:
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Posts analysés", total)
    col2.metric("Fiables", fiable, delta=f"{fiable / total * 100:.0f}%")
    col3.metric("Douteux", douteux, delta=f"{douteux / total * 100:.0f}%", delta_color="off")
    col4.metric("Fake News", fake, delta=f"{fake / total * 100:.0f}%", delta_color="inverse")

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
        label = post["credibility"]["label"]
        color = LABEL_COLORS[label]
        with st.expander(f"#{i} — @{post['author_handle']} — :{color}[{label.upper()}] ({score:.0%})"):
            st.markdown(f"**Texte original :** {post['text']}")
            st.markdown(f"**Émotion dominante :** {post['emotion']['dominant_emotion']}")
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
            "Confiance": f"{r['credibility']['confidence']:.0%}",
            "Émotion": r["emotion"]["dominant_emotion"],
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

    st.dataframe(
        df_filtered,
        use_container_width=True,
        column_config={
            "Label": st.column_config.TextColumn("Label", width="small"),
            "Confiance": st.column_config.TextColumn("Confiance", width="small"),
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
        search_query = st.text_input("Recherche", placeholder="Ex: élections, santé, vaccin...")
        lang_options = {"Toutes": None, "Français": "fr", "English": "en"}
        lang_filter = st.selectbox("Langue", list(lang_options.keys()))
        num_posts = st.slider("Nombre de posts", 10, 200, 50)
        save_to_db = st.checkbox("Sauvegarder en BDD", value=False)
        analyze_btn = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    # Lancement de l'analyse
    if analyze_btn:
        if not handle or not password:
            st.error("Renseigne tes identifiants Bluesky dans la sidebar.")
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
        tab_overview, tab_details, tab_emotions, tab_energy, tab_history = st.tabs(
            ["Vue d'ensemble", "Détails", "Émotions", "Green IT", "Historique"]
        )
        with tab_overview:
            st.info("Lance une analyse depuis la sidebar pour voir les résultats.")
        with tab_history:
            render_history()
        return

    tab_overview, tab_details, tab_emotions, tab_energy, tab_history = st.tabs(
        ["Vue d'ensemble", "Détails", "Émotions", "Green IT", "Historique"]
    )

    with tab_overview:
        render_overview(results)

    with tab_details:
        render_details(results)

    with tab_emotions:
        render_emotions(results)

    with tab_energy:
        render_energy(energy)

    with tab_history:
        render_history()


if __name__ == "__main__":
    main()
