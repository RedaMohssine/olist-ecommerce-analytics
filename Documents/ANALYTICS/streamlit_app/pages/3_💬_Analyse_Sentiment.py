"""
💬 Page Analyse de Sentiment des Avis
"""

import streamlit as st
from components.style import load_global_styles
from components.topbar import render_topbar
from components.sidebar import render_sidebar
import pandas as pd
import numpy as np
from datetime import datetime
import re
import nltk
from components.auth import require_admin
from components.translations import get_text
from components.charts import create_bar_chart, create_pie_chart, create_kpi_chart, create_line_chart
from utils.data_loader import load_reviews, load_orders, load_products, load_order_items, load_sellers
from utils.model_manager import ModelManager, load_sentiment_model

# Vérification des droits admin
require_admin()

st.set_page_config(page_title="Analyse Sentiment", page_icon="💬", layout="wide")

# CSS global
load_global_styles()

# Topbar login/logout
render_topbar(key_prefix="sent_")

# Sidebar menu
render_sidebar()

# ========================================
# PREPROCESSING FUNCTION (IDENTIQUE AU NOTEBOOK)
# ========================================
# Télécharger les stopwords si nécessaire
try:
    from nltk.corpus import stopwords
    stopwords_pt = set(stopwords.words('portuguese'))
except:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stopwords_pt = set(stopwords.words('portuguese'))

def preprocess_text(text):
    """
    Préprocessing NLP IDENTIQUE au notebook.
    Étapes :
    1. Minuscules
    2. Nettoyage (conservation des lettres accentuées + espaces)
    3. Suppression des stopwords
    4. Filtrage : tokens courts, espaces
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Minuscules
    text = text.lower()
    
    # Nettoyage : on garde lettres (avec accents), chiffres, espaces et !?
    text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ0-9\s!?]', ' ', text)
    
    # Réduction des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenisation simple et filtrage
    tokens = []
    for word in text.split():
        # Supprimer stopwords, ponctuation pure, et mots très courts
        if (word not in stopwords_pt 
            and word not in ['!', '?', '.', ',', '...']
            and len(word) >= 2):
            tokens.append(word)
    
    return " ".join(tokens)

# Vérification des droits admin
require_admin()

st.set_page_config(page_title="Analyse Sentiment", page_icon="💬", layout="wide")

# CSS Custom
st.markdown("""
<style>
    .sentiment-positive {
        background: linear-gradient(135deg, #00d084 0%, #00b368 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #ffd93d 0%, #ffc107 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-meter {
        background: rgba(255, 255, 255, 0.2);
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 1rem;
    }
    .confidence-fill {
        height: 100%;
        background: white;
        transition: width 0.3s ease;
    }
    .review-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #009739;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='admin-header'>
    <h1 style='margin: 0;'>💬 Analyse de Sentiment</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
        Classification automatique des avis clients
    </p>
</div>
""", unsafe_allow_html=True)

# ========================================
# CHARGEMENT DU MODÈLE
# ========================================
model, vectorizer = load_sentiment_model()
model_manager = ModelManager('sentiment')
metadata = model_manager.get_metadata()

# Vérifier si le modèle est chargé
if model is None:
    st.error("❌ Aucun modèle de sentiment n'est chargé. Veuillez entraîner et sauvegarder un modèle depuis le notebook.")
    st.stop()

if vectorizer is None:
    st.warning("⚠️ Vectoriseur TF-IDF non trouvé. Les prédictions peuvent être incorrectes.")

# ========================================
# INDICATEURS DU MODÈLE
# ========================================
st.markdown("## 📊 Performance du Modèle")

if metadata:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = metadata.get('accuracy', 0)
        st.markdown(create_kpi_chart(
            "Accuracy",
            f"{accuracy:.2%}",
            "🎯"
        ), unsafe_allow_html=True)
    
    with col2:
        f1_score = metadata.get('f1_score', metadata.get('mae', 0))
        st.markdown(create_kpi_chart(
            "F1-Score",
            f"{f1_score:.2%}" if f1_score > 1 else f"{f1_score:.3f}",
            "📊"
        ), unsafe_allow_html=True)
    
    with col3:
        upload_date = metadata.get('upload_date', 'N/A')
        if upload_date != 'N/A':
            upload_date = datetime.fromisoformat(upload_date).strftime('%d/%m/%Y')
        st.markdown(create_kpi_chart(
            "Dernière MAJ",
            upload_date,
            "📅"
        ), unsafe_allow_html=True)
    
    with col4:
        author = metadata.get('author', 'System')
        st.markdown(create_kpi_chart(
            "Auteur",
            author,
            "👤"
        ), unsafe_allow_html=True)
else:
    st.warning("⚠️ Métadonnées du modèle non disponibles")

st.markdown("---")

# ========================================
# MODE DE PRÉDICTION
# ========================================
st.markdown("## 🎯 Mode d'Analyse")

mode = st.radio(
    "Choisissez le mode",
    ["🔮 Analyse Unique", "📊 Analyse par Lot", "📈 Dashboard Sentiments", "🏪 Analyse par Vendeur"],
    horizontal=True
)

# ========================================
# MODE 1: ANALYSE UNIQUE
# ========================================
if mode == "🔮 Analyse Unique":
    st.markdown("### 🔮 Analyse d'un Avis Client")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 📝 Saisir l'Avis")
        
        # Exemples d'avis (afficher AVANT le widget pour pouvoir modifier la valeur par défaut)
        with st.expander("💡 Charger un exemple"):
            examples = {
                "Positif": "Produto excelente! Entrega rápida e embalagem perfeita. Recomendo!",
                "Négatif": "Muito decepcionado. O produto chegou com defeito e o atendimento foi péssimo.",
                "Neutre": "Produto ok, nada de especial. Chegou no prazo."
            }
            
            col_ex1, col_ex2, col_ex3 = st.columns(3)
            
            with col_ex1:
                if st.button("😊 Positif", key="ex_positif", width='stretch'):
                    st.session_state['example_text'] = examples["Positif"]
            
            with col_ex2:
                if st.button("😐 Neutre", key="ex_neutre", width='stretch'):
                    st.session_state['example_text'] = examples["Neutre"]
            
            with col_ex3:
                if st.button("😞 Négatif", key="ex_negatif", width='stretch'):
                    st.session_state['example_text'] = examples["Négatif"]
        
        # Widget text_area avec valeur par défaut depuis session_state
        default_text = st.session_state.get('example_text', '')
        review_text = st.text_area(
            "Texte de l'avis",
            value=default_text,
            height=200,
            placeholder="Entrez l'avis client à analyser...",
            key='review_text'
        )
        
        col_meta1, col_meta2 = st.columns(2)
        
        with col_meta1:
            review_score = st.slider(
                "Note client (1-5)",
                min_value=1,
                max_value=5,
                value=3,
                help="Note attribuée par le client (optionnel)"
            )
        
        with col_meta2:
            review_date = st.date_input(
                "Date de l'avis",
                value=datetime.now()
            )
        
        # Bouton d'analyse
        if st.button("🚀 Analyser le Sentiment", type="primary", width='stretch'):
            if review_text.strip():
                with st.spinner("🔄 Analyse en cours..."):
                    try:
                        # Prétraiter le texte (IDENTIQUE AU NOTEBOOK)
                        processed_text = preprocess_text(review_text)
                        
                        # Vectoriser le texte avec TF-IDF
                        if vectorizer is not None and model is not None:
                            X_vectorized = vectorizer.transform([processed_text])
                            
                            # Prédire avec le modèle
                            prediction = model.predict(X_vectorized)[0]
                            proba = model.predict_proba(X_vectorized)[0]
                            
                            # Les classes du modèle sklearn sont en ordre alphabétique
                            classes = model.classes_  # ['negative', 'neutral', 'positive']
                            
                            # Créer un dictionnaire de probabilités par classe
                            proba_dict = {cls: prob for cls, prob in zip(classes, proba)}
                            
                            # La prédiction est déjà le label textuel (pas un index)
                            sentiment_class = prediction
                            
                            # Mapping pour l'affichage
                            sentiment_display = {
                                "negative": "Négatif",
                                "neutral": "Neutre",
                                "positive": "Positif"
                            }
                            sentiment = sentiment_display[sentiment_class]
                            confidence = float(proba_dict[sentiment_class])
                        else:
                            # Fallback: analyse basique par mots-clés
                            positive_words = ['excelente', 'ótimo', 'perfeito', 'recomendo', 'bom', 'rápida', 'adorei']
                            negative_words = ['péssimo', 'ruim', 'defeito', 'decepcionado', 'problema', 'horrível']
                            
                            text_lower = review_text.lower()
                            pos_count = sum(1 for word in positive_words if word in text_lower)
                            neg_count = sum(1 for word in negative_words if word in text_lower)
                            
                            if pos_count > neg_count:
                                sentiment = "Positif"
                                confidence = 0.7 + (pos_count * 0.05)
                                sentiment_class = "positive"
                            elif neg_count > pos_count:
                                sentiment = "Négatif"
                                confidence = 0.7 + (neg_count * 0.05)
                                sentiment_class = "negative"
                            else:
                                sentiment = "Neutre"
                                confidence = 0.6 + np.random.rand() * 0.1
                                sentiment_class = "neutral"
                        
                        confidence = min(0.99, confidence)
                        
                        # Stocker TOUS les détails pour debugging
                        st.session_state['sentiment_result'] = {
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'class': sentiment_class,
                            'text': review_text,
                            'processed_text': processed_text,
                            'score': review_score,
                            'proba_dict': proba_dict if vectorizer is not None else None
                        }
                    
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
            else:
                st.warning("⚠️ Veuillez saisir un avis à analyser")
    
    with col2:
        st.markdown("#### 📊 Résultat de l'Analyse")
        
        if 'sentiment_result' in st.session_state:
            result = st.session_state['sentiment_result']
            
            sentiment = result['sentiment']
            confidence = result['confidence']
            sentiment_class = result['class']
            
            # Affichage du résultat
            if sentiment_class == "positive":
                st.markdown(f"""
                <div class='sentiment-positive'>
                    <h2 style='margin: 0;'>😊 {sentiment}</h2>
                    <p style='font-size: 1.2rem; margin: 1rem 0;'>Avis Positif</p>
                    <div class='confidence-meter'>
                        <div class='confidence-fill' style='width: {confidence*100}%;'></div>
                    </div>
                    <p style='margin: 0.5rem 0 0 0;'>Confiance: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif sentiment_class == "negative":
                st.markdown(f"""
                <div class='sentiment-negative'>
                    <h2 style='margin: 0;'>😞 {sentiment}</h2>
                    <p style='font-size: 1.2rem; margin: 1rem 0;'>Avis Négatif</p>
                    <div class='confidence-meter'>
                        <div class='confidence-fill' style='width: {confidence*100}%;'></div>
                    </div>
                    <p style='margin: 0.5rem 0 0 0;'>Confiance: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown(f"""
                <div class='sentiment-neutral'>
                    <h2 style='margin: 0;'>😐 {sentiment}</h2>
                    <p style='font-size: 1.2rem; margin: 1rem 0;'>Avis Neutre</p>
                    <div class='confidence-meter'>
                        <div class='confidence-fill' style='width: {confidence*100}%; background: #333;'></div>
                    </div>
                    <p style='margin: 0.5rem 0 0 0;'>Confiance: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Détails
            st.markdown("##### 📋 Détails")
            
            # Afficher les probabilités par classe si disponibles
            if result.get('proba_dict'):
                st.markdown("**Probabilités par classe:**")
                for cls in ['negative', 'neutral', 'positive']:
                    emoji_map = {'negative': '😞', 'neutral': '😐', 'positive': '😊'}
                    prob = result['proba_dict'].get(cls, 0)
                    st.write(f"{emoji_map[cls]} {cls.capitalize()}: {prob:.1%}")
            
            st.info(f"""
            **Note client:** {result['score']}/5 ⭐
            
            **Longueur:** {len(result['text'])} caractères
            
            **Mots:** {len(result['text'].split())} mots
            """)
            
            # Afficher le texte préprocessé
            with st.expander("🔍 Voir le texte préprocessé"):
                st.text(result.get('processed_text', 'N/A'))
            
            # Action recommandée
            st.markdown("##### 💡 Action Recommandée")
            if sentiment_class == "positive":
                st.success("✅ Encourager le client à laisser plus d'avis")
            elif sentiment_class == "negative":
                st.error("⚠️ Contacter le client pour résoudre le problème")
            else:
                st.info("ℹ️ Surveiller l'évolution de la satisfaction")
        
        else:
            st.info("👈 Saisissez un avis et cliquez sur 'Analyser le Sentiment'")
            
            st.markdown("##### 🎯 Comment ça marche ?")
            st.markdown("""
            1. **Saisir** un avis client
            2. **Optionnel:** Ajouter la note
            3. **Analyser** avec le ML
            4. **Obtenir** le sentiment + confiance
            5. **Agir** selon le résultat
            """)

# ========================================
# MODE 2: ANALYSE PAR LOT
# ========================================
elif mode == "📊 Analyse par Lot":
    st.markdown("### 📊 Analyse par Lot (CSV)")
    
    st.info("""
    **Format CSV requis:**
    - review_id (optionnel)
    - review_text (obligatoire)
    - review_score (optionnel)
    """)
    
    uploaded_file = st.file_uploader("📁 Upload fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            if 'review_text' not in df_batch.columns:
                st.error("❌ Colonne 'review_text' manquante dans le CSV")
            else:
                st.success(f"✅ {len(df_batch)} avis chargés")
                
                st.dataframe(df_batch.head(), width='stretch')
                
                if st.button("🚀 Analyser Tous les Avis", type="primary"):
                    with st.spinner("🔄 Analyse en cours..."):
                        sentiments = []
                        confidences = []
                        raw_labels = []

                        texts = df_batch['review_text'].fillna('').astype(str).tolist()
                        processed = [preprocess_text(t) for t in texts]

                        if vectorizer is not None and model is not None:
                            X = vectorizer.transform(processed)
                            preds = model.predict(X)
                            probas = model.predict_proba(X)
                            classes = list(model.classes_)

                            display_map = {
                                'positive': 'Positif',
                                'neutral': 'Neutre',
                                'negative': 'Négatif'
                            }

                            for i, pred in enumerate(preds):
                                raw_labels.append(pred)
                                sentiments.append(display_map.get(pred, pred))
                                if pred in classes:
                                    conf = float(probas[i][classes.index(pred)])
                                else:
                                    conf = float(max(probas[i]))
                                confidences.append(conf)
                        else:
                            # Fallback basé sur review_score si pas de modèle
                            scores = df_batch['review_score'] if 'review_score' in df_batch.columns else pd.Series([3] * len(df_batch))
                            for score in scores:
                                if score >= 4:
                                    raw_labels.append('positive')
                                    sentiments.append('Positif')
                                    confidences.append(0.75)
                                elif score <= 2:
                                    raw_labels.append('negative')
                                    sentiments.append('Négatif')
                                    confidences.append(0.70)
                                else:
                                    raw_labels.append('neutral')
                                    sentiments.append('Neutre')
                                    confidences.append(0.60)

                        df_batch['predicted_label'] = raw_labels
                        df_batch['predicted_sentiment'] = sentiments
                        df_batch['confidence'] = confidences
                        
                        st.success("✅ Analyse terminée!")
                        
                        # Statistiques
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            positive_pct = (df_batch['predicted_sentiment'] == 'Positif').mean()
                            st.metric("😊 Positifs", f"{positive_pct:.1%}")
                        
                        with col2:
                            neutral_pct = (df_batch['predicted_sentiment'] == 'Neutre').mean()
                            st.metric("😐 Neutres", f"{neutral_pct:.1%}")
                        
                        with col3:
                            negative_pct = (df_batch['predicted_sentiment'] == 'Négatif').mean()
                            st.metric("😞 Négatifs", f"{negative_pct:.1%}")
                        
                        # Distribution
                        sentiment_counts = df_batch['predicted_sentiment'].value_counts()
                        chart = create_pie_chart(
                            pd.DataFrame({
                                'Sentiment': sentiment_counts.index,
                                'Count': sentiment_counts.values
                            }),
                            'Sentiment',
                            'Count',
                            "Distribution des Sentiments"
                        )
                        st.plotly_chart(chart, width='stretch')
                        
                        # Résultats
                        st.markdown("#### 📋 Résultats Détaillés")
                        st.dataframe(df_batch, width='stretch')
                        
                        # Export
                        csv = df_batch.to_csv(index=False)
                        st.download_button(
                            "💾 Télécharger les Résultats",
                            csv,
                            "sentiment_analysis_results.csv",
                            "text/csv",
                            width='stretch'
                        )
        
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

# ========================================
# MODE 3: DASHBOARD SENTIMENTS
# ========================================
elif mode == "📈 Dashboard Sentiments":
    st.markdown("### 📈 Dashboard des Sentiments Globaux")
    
    reviews = load_reviews()
    
    if reviews is not None:
        # Classification basée sur le score
        reviews['sentiment'] = reviews['review_score'].apply(
            lambda x: 'Positif' if x >= 4 else ('Négatif' if x <= 2 else 'Neutre')
        )
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reviews = len(reviews)
            st.metric("Total Avis", f"{total_reviews:,}")
        
        with col2:
            positive_pct = (reviews['sentiment'] == 'Positif').mean()
            st.metric("😊 Positifs", f"{positive_pct:.1%}")
        
        with col3:
            negative_pct = (reviews['sentiment'] == 'Négatif').mean()
            st.metric("😞 Négatifs", f"{negative_pct:.1%}")
        
        with col4:
            avg_score = reviews['review_score'].mean()
            st.metric("Note Moyenne", f"{avg_score:.2f}/5")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des sentiments
            sentiment_counts = reviews['sentiment'].value_counts()
            chart = create_pie_chart(
                pd.DataFrame({
                    'Sentiment': sentiment_counts.index,
                    'Count': sentiment_counts.values
                }),
                'Sentiment',
                'Count',
                "Répartition des Sentiments"
            )
            st.plotly_chart(chart, width='stretch')
        
        with col2:
            # Distribution des notes
            score_counts = reviews['review_score'].value_counts().sort_index()
            chart2 = create_bar_chart(
                pd.DataFrame({
                    'Note': score_counts.index,
                    'Nombre': score_counts.values
                }),
                'Note',
                'Nombre',
                "Distribution des Notes"
            )
            st.plotly_chart(chart2, width='stretch')
        
        # Évolution temporelle
        st.markdown("#### 📊 Évolution Temporelle")
        
        reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
        monthly_sentiment = reviews.set_index('review_creation_date').resample('ME')['review_score'].mean()
        
        chart3 = create_line_chart(
            pd.DataFrame({
                'Mois': monthly_sentiment.index,
                'Note Moyenne': monthly_sentiment.values
            }),
            'Mois',
            'Note Moyenne',
            "Évolution de la Satisfaction"
        )
        st.plotly_chart(chart3, width='stretch')
        
        # Avis négatifs récents
        st.markdown("#### ⚠️ Avis Négatifs Récents (à traiter)")
        
        negative_reviews = reviews[reviews['sentiment'] == 'Négatif'].sort_values(
            'review_creation_date', ascending=False
        ).head(10)
        
        for idx, row in negative_reviews.iterrows():
            comment = row.get('review_comment_message', 'Pas de commentaire')
            if pd.isna(comment):
                comment = "Pas de commentaire"
            
            st.markdown(f"""
            <div class='review-card'>
                <strong>⭐ {row['review_score']}/5</strong> - {row['review_creation_date'].strftime('%d/%m/%Y')}
                <p style='margin: 0.5rem 0;'>{comment[:200]}...</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Données non disponibles")

# ========================================
# MODE 4: ANALYSE PAR VENDEUR
# ========================================
elif mode == "🏪 Analyse par Vendeur":
    st.markdown("### 🏪 Analyse des Sentiments par Vendeur")
    
    reviews = load_reviews()
    order_items = load_order_items()
    sellers = load_sellers()
    
    if reviews is not None and order_items is not None and sellers is not None:
        # Merge reviews avec order_items pour avoir seller_id
        reviews_items = reviews.merge(order_items[['order_id', 'seller_id']], on='order_id', how='inner')
        
        # Classification
        reviews_items['sentiment'] = reviews_items['review_score'].apply(
            lambda x: 'Positif' if x >= 4 else ('Négatif' if x <= 2 else 'Neutre')
        )
        
        # Agrégation par vendeur
        seller_sentiments = reviews_items.groupby('seller_id').agg({
            'review_id': 'count',
            'review_score': 'mean'
        }).reset_index()
        seller_sentiments.columns = ['seller_id', 'n_reviews', 'avg_score']
        
        st.markdown("#### 📊 Top/Flop Vendeurs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("##### 🏆 Top 5 Vendeurs")
            top5 = seller_sentiments.sort_values(['avg_score', 'n_reviews'], ascending=[False, False]).head(5)
            st.dataframe(top5, hide_index=True, width='stretch')
        
        with col2:
            st.error("##### ⚠️ Vendeurs à Surveiller")
            bottom5 = seller_sentiments.sort_values(['avg_score', 'n_reviews'], ascending=[True, False]).head(5)
            st.dataframe(bottom5, hide_index=True, width='stretch')
    
    else:
        st.error("❌ Données non disponibles")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>💬 Analyse de Sentiment</p>
    <p>💡 Astuce: Les avis négatifs sont une opportunité d'amélioration</p>
</div>
""", unsafe_allow_html=True)
