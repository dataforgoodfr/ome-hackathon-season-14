

import streamlit as st
import pandas as pd
import plotly.express as px

# Import de votre logique métier existante
# On suppose que le fichier final/orchestrator.py contient la fonction predict_text
# telle que vous l'avez fournie.
try:
    from final.orchestrator import predict_text
except ImportError as e:
    st.error(f"Erreur d'importation : {e}. Assurez-vous de lancer l'app depuis la racine du projet.")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="Hackathon Pipeline NLP",
    page_icon="",
    layout="wide"
)

# --- CSS Personnalisé pour l'esthétique ---
st.markdown("""
<style>
    .segment-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 5px solid #ccc;
    }
    .cat-agriculture { border-left-color: #2ecc71; background-color: #eafaf1; }
    .cat-transport { border-left-color: #3498db; background-color: #ebf5fb; }
    .cat-energie { border-left-color: #f1c40f; background-color: #fef9e7; }
    .cat-autre { border-left-color: #95a5a6; background-color: #f4f6f6; }
</style>
""", unsafe_allow_html=True)

# --- En-tête ---
st.title(" Visualisation du Pipeline de Classification")
st.markdown("""
**Objectif :** Transformer un reportage brut en segments classifiés.
1. **Input** : Texte brut.
2. **Segmentation** : Découpage intelligent.
3. **Prédiction** : Agriculture, Transport, Énergie ou Autre.
""")

st.divider()

# --- Zone 1 : Input (Entrée) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Entrée (Input)")

    # Tentative de chargement du fichier par défaut
    default_text = ""
    try:
        with open("final/input_text.txt", "r", encoding="utf-8") as f:
            default_text = f.read()
    except FileNotFoundError:
        default_text = "Le fichier input_text.txt est introuvable. Collez votre texte ici."

    input_text = st.text_area(
        "Reportage à analyser :",
        value=default_text,
        height=400,
        help="Modifiez ce texte pour tester différents scénarios."
    )

    analyze_btn = st.button("Lancer l'analyse", type="primary", use_container_width=True)

# --- Zone 2 & 3 : Traitement et Résultats ---
if analyze_btn:
    with col2:
        with st.spinner("Segmentation et classification en cours..."):
            try:
                # Appel à votre orchestrateur
                result = predict_text(input_text)

                # --- Métriques Globales ---
                st.subheader("2. Métriques de Nettoyage")
                m1, m2, m3 = st.columns(3)
                m1.metric("Mots (Brut)", result.initial_word_count)
                m2.metric("Mots (Nettoyé)", result.final_word_count)
                m3.metric("Doublons retirés", result.nb_duplicates)

                st.divider()

                # --- Préparation des données pour la visualisation ---
                # On transforme la liste de segments en DataFrame pour faciliter les graphiques
                data = []
                for s in result.segments:
                    data.append({
                        "Category": s.category,
                        "Word Count": s.word_count,
                        "Confidence": s.score,
                        "Content": s.content
                    })
                df_segments = pd.DataFrame(data)

                # --- Visualisation : Répartition ---
                st.subheader("3. Résultat : Répartition Thématique")

                if not df_segments.empty:
                    # Graphique Camembert (Donut) avec Plotly
                    fig = px.pie(
                        df_segments,
                        names='Category',
                        values='Word Count',
                        title='Proportion du texte par thématique (basé sur le nombre de mots)',
                        hole=0.4,
                        color='Category',
                        color_discrete_map={
                            'Agriculture': '#2ecc71',
                            'Transport': '#3498db',
                            'Energie': '#f1c40f',
                            'Autre': '#95a5a6'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Aucun segment valide trouvé.")

                st.divider()

                # --- Visualisation : Détail des Segments ---
                st.subheader("4. Détail de la Segmentation")
                st.info("Ci-dessous, le texte découpé en blocs sémantiques avec leur prédiction.")

                for seg in result.segments:
                    # Choix de la classe CSS pour la couleur
                    css_class = "cat-autre"
                    if "agri" in seg.category.lower(): css_class = "cat-agriculture"
                    elif "trans" in seg.category.lower(): css_class = "cat-transport"
                    elif "ener" in seg.category.lower(): css_class = "cat-energie"

                    # Affichage du segment
                    st.markdown(f"""
                    <div class="segment-box {css_class}">
                        <strong>{seg.category}</strong> <small>(Confiance: {seg.score:.2f} | Mots: {seg.word_count})</small><br>
                        <hr style="margin: 5px 0; opacity: 0.2">
                        {seg.content}
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'analyse : {e}")
                # Utile pour le débogage pendant le hackathon
                st.exception(e)

elif not analyze_btn:
    with col2:
        st.info("Cliquez sur **Lancer l'analyse** pour voir le pipeline en action.")
