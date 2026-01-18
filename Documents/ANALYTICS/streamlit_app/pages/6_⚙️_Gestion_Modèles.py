"""
⚙️ Page Gestion des Modèles ML - Upload, Versioning, et Configuration
"""

import streamlit as st
from components.style import load_global_styles
from components.topbar import render_topbar
from components.sidebar import render_sidebar
import joblib
from datetime import datetime
from pathlib import Path
from components.auth import require_admin
from utils.model_manager import ModelManager
import json

# Vérification des droits admin
require_admin()

st.set_page_config(page_title="Gestion Modèles", page_icon="⚙️", layout="wide")

# CSS global
load_global_styles()

# Topbar login/logout
render_topbar(key_prefix="models_")

# Sidebar menu
render_sidebar()

# Header
st.markdown("""
<div class='admin-header'>
    <h1 style='margin: 0;'>⚙️ Gestion des Modèles ML</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
        Upload, versioning et configuration des modèles de Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ========================================
# VUE D'ENSEMBLE DES MODÈLES
# ========================================
st.markdown("## 📊 État des Modèles")

models_status = ModelManager.get_all_models_status()

col1, col2, col3, col4 = st.columns(4)

model_types = {
    'shipping': ('🚚', 'Livraison'),
    'sentiment': ('💬', 'Sentiment'),
    'orders': ('📦', 'Commandes'),
    'clustering': ('🎯', 'Recommandation')
}

MODEL_DIR_MAP = {
    'shipping': Path("shipping_forecast"),
    'orders': Path("orders_forecast"),
    'clustering': Path("recommendation"),
    'sentiment': Path("sentiment")
}

def get_model_assets(model_type):
    base_dir = Path(__file__).parent.parent / "models" / MODEL_DIR_MAP.get(model_type, Path(model_type))
    assets = {
        'base_dir': base_dir,
        'config': base_dir / "config.json",
        'feature_names': base_dir / "feature_names.pkl",
        'features_info': base_dir / "features_info.json"
    }
    return assets

for idx, (model_type, (emoji, name)) in enumerate(model_types.items()):
    status = models_status[model_type]
    col = [col1, col2, col3, col4][idx]
    
    with col:
        if status['loaded']:
            st.success(f"{emoji} **{name}**\n\n✅ Chargé")
            
            metadata = status['metadata']
            if metadata:
                upload_date = metadata.get('upload_date') or metadata.get('trained_date')
                if upload_date:
                    try:
                        dt = datetime.fromisoformat(upload_date)
                        st.caption(f"📅 {dt.strftime('%d/%m/%Y')}")
                    except Exception:
                        st.caption(f"📅 {upload_date}")
                
                if 'metrics' in metadata:
                    metrics = metadata.get('metrics', {})
                    if 'r2_score' in metrics:
                        st.caption(f"🎯 R²: {metrics['r2_score']:.3f}")
                    if 'mae' in metrics:
                        st.caption(f"📊 MAE: {metrics['mae']:.2f}")
                elif 'accuracy' in metadata:
                    st.caption(f"🎯 Accuracy: {metadata['accuracy']:.2%}")
                elif 'mae' in metadata:
                    st.caption(f"📊 MAE: {metadata['mae']:.2f}")
            else:
                st.caption("📦 Modèle par défaut")
        else:
            st.warning(f"{emoji} **{name}**\n\n⚠️ Non chargé")
        
        st.caption(f"🗂️ {status['history_count']} versions")

st.markdown("---")

# ========================================
# CHARGEMENT CONFIGURATION
# ========================================
CONFIG_PATH = Path(__file__).parent.parent / "config" / "models_config.json"

def load_app_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_app_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

app_config = load_app_config()
app_config.setdefault('model_settings', {})

# ========================================
# SÉLECTION DU MODÈLE À GÉRER
# ========================================
st.markdown("## 🎛️ Gestion Individuelle")

selected_model = st.selectbox(
    "Sélectionnez un modèle à gérer",
    options=list(model_types.keys()),
    format_func=lambda x: f"{model_types[x][0]} {model_types[x][1]}",
    key='model_selector'
)

model_manager = ModelManager(selected_model)

st.markdown(f"### {model_types[selected_model][0]} Modèle: {model_types[selected_model][1]}")

# Tabs pour les différentes actions
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Nouveau Modèle",
    "📋 Informations Actuelles",
    "🗂️ Historique",
    "⚙️ Configuration"
])

# ========================================
# TAB 1: UPLOAD NOUVEAU MODÈLE
# ========================================
with tab1:
    st.markdown("#### 📤 Téléverser un Nouveau Modèle")
    
    st.info("""
    **Instructions:**
    1. Entraînez votre modèle dans un notebook
    2. Sauvegardez-le avec `joblib.dump(model, 'model.joblib')`
    3. Uploadez le fichier `.joblib` ci-dessous
    4. Ajoutez les métadonnées (optionnel mais recommandé)
    5. Validez pour remplacer le modèle en production
    
    **📦 Modèles par défaut:** Les notebooks d'entraînement génèrent automatiquement 
    des modèles par défaut. Vous pouvez les remplacer avec vos propres versions ici.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload du fichier
        uploaded_file = st.file_uploader(
            "Sélectionnez un fichier .joblib",
            type=['joblib', 'pkl'],
            key=f'upload_{selected_model}'
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Fichier chargé: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
            
            # Métadonnées
            st.markdown("##### 📝 Métadonnées du Modèle")
            
            col_meta1, col_meta2 = st.columns(2)
            
            with col_meta1:
                accuracy = st.number_input(
                    "Accuracy / R²",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    key=f'accuracy_{selected_model}'
                )
                
                mae = st.number_input(
                    "MAE (Mean Absolute Error)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    key=f'mae_{selected_model}'
                )
            
            with col_meta2:
                author = st.text_input(
                    "Auteur",
                    placeholder="Nom de l'auteur",
                    key=f'author_{selected_model}'
                )
                
                description = st.text_area(
                    "Description",
                    placeholder="Décrivez les améliorations ou changements",
                    key=f'desc_{selected_model}'
                )
            
            # Validation et upload
            st.markdown("---")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("✅ Valider et Remplacer le Modèle", 
                           type="primary", 
                           width='stretch',
                           key=f'validate_{selected_model}'):
                    
                    with st.spinner("🔄 Sauvegarde en cours..."):
                        try:
                            # Charger le modèle depuis le fichier uploadé
                            model = joblib.load(uploaded_file)
                            
                            # Préparer les métadonnées
                            metadata = {
                                'accuracy': accuracy,
                                'mae': mae,
                                'author': author,
                                'description': description,
                                'filename': uploaded_file.name
                            }
                            
                            # Sauvegarder
                            success = model_manager.save_model(model, metadata)
                            
                            if success:
                                st.success("✅ Modèle sauvegardé avec succès!")
                                st.balloons()
                                
                                # Effacer le cache pour recharger
                                st.cache_resource.clear()
                                
                                st.info("🔄 Le modèle sera actif au prochain rechargement de la page")
                            else:
                                st.error("❌ Erreur lors de la sauvegarde")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur: {str(e)}")
            
            with col_btn2:
                if st.button("🔍 Tester le Modèle", 
                           width='stretch',
                           key=f'test_{selected_model}'):
                    try:
                        model = joblib.load(uploaded_file)
                        st.success("✅ Modèle chargé avec succès - Prêt à être déployé")
                        st.info(f"Type: {type(model).__name__}")
                    except Exception as e:
                        st.error(f"❌ Erreur lors du chargement: {str(e)}")
    
    with col2:
        st.markdown("##### 💡 Conseils")
        st.markdown("""
        **Format supporté:**
        - `.joblib` (recommandé)
        - `.pkl` (pickle)
        
        **Bonnes pratiques:**
        - Testez le modèle avant upload
        - Documentez les changements
        - Vérifiez les métriques
        - Gardez l'historique
        
        **Sécurité:**
        - L'ancien modèle est archivé
        - Possibilité de restaurer
        - Versioning automatique
        """)

# ========================================
# TAB 2: INFORMATIONS ACTUELLES
# ========================================
with tab2:
    st.markdown("#### 📋 Modèle Actif")
    
    model = model_manager.load_model()
    metadata = model_manager.get_metadata()
    
    if model is not None:
        st.success("✅ Un modèle est actuellement chargé")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔧 Informations Techniques")
            st.json({
                "Type": type(model).__name__,
                "Module": model.__class__.__module__,
            })
        
        with col2:
            st.markdown("##### 📊 Métadonnées")
            if metadata:
                st.json(metadata)
            else:
                st.info("Aucune métadonnée disponible")
        
        # Afficher les attributs du modèle si possible
        with st.expander("🔍 Détails du Modèle"):
            try:
                assets = get_model_assets(selected_model)
                if assets['feature_names'].exists():
                    import pickle
                    with open(assets['feature_names'], 'rb') as f:
                        feature_names = pickle.load(f)
                    st.write("**Features (liste):**")
                    st.dataframe(pd.DataFrame({'feature': feature_names}))
                elif assets['features_info'].exists():
                    with open(assets['features_info'], 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    st.write("**Features Info:**")
                    st.json(info)
                
                if hasattr(model, 'get_params'):
                    st.write("**Paramètres:**")
                    st.json(model.get_params())
                
                if hasattr(model, 'feature_importances_'):
                    st.write("**Top 10 Features:**")
                    import pandas as pd
                    features = pd.DataFrame({
                        'feature': [f"feature_{i}" for i in range(len(model.feature_importances_))],
                        'importance': model.feature_importances_
                    }).nlargest(10, 'importance')
                    st.dataframe(features)
            except:
                st.info("Détails non disponibles pour ce type de modèle")
    
    else:
        st.warning("⚠️ Aucun modèle n'est actuellement chargé")
        st.info("Uploadez un nouveau modèle dans l'onglet 'Upload Nouveau Modèle'")

# ========================================
# TAB 3: HISTORIQUE
# ========================================
with tab3:
    st.markdown("#### 🗂️ Historique des Versions")
    
    history = model_manager.get_history()
    
    if history:
        st.info(f"📚 {len(history)} versions archivées")
        
        for i, filename in enumerate(history):
            with st.expander(f"Version {i+1}: {filename}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(f"📁 {filename}")
                    # Extraire la date du filename
                    try:
                        date_str = filename.split('_')[1].split('.')[0]
                        date_obj = datetime.strptime(date_str, "%Y%m%d")
                        st.caption(f"📅 {date_obj.strftime('%d/%m/%Y')}")
                    except:
                        pass
                
                with col2:
                    if st.button("🔄 Restaurer", 
                               key=f"restore_{filename}"):
                        if st.session_state.get(f'confirm_restore_{filename}'):
                            success = model_manager.restore_from_history(filename)
                            if success:
                                st.success("✅ Modèle restauré!")
                                st.cache_resource.clear()
                                st.rerun()
                        else:
                            st.session_state[f'confirm_restore_{filename}'] = True
                            st.warning("⚠️ Cliquez à nouveau pour confirmer")
                
                with col3:
                    if st.button("🗑️ Supprimer", 
                               key=f"delete_{filename}"):
                        if st.session_state.get(f'confirm_delete_{filename}'):
                            success = model_manager.delete_from_history(filename)
                            if success:
                                st.success("✅ Version supprimée!")
                                st.rerun()
                        else:
                            st.session_state[f'confirm_delete_{filename}'] = True
                            st.warning("⚠️ Cliquez à nouveau pour confirmer")
    else:
        st.info("📭 Aucune version archivée pour l'instant")

# ========================================
# TAB 4: CONFIGURATION
# ========================================
with tab4:
    st.markdown("#### ⚙️ Configuration du Modèle")
    
    # Configuration spécifique selon le type de modèle (persistée)
    model_settings = app_config['model_settings'].get(selected_model, {})
    
    if selected_model == 'shipping':
        st.markdown("##### 🚚 Paramètres de Prédiction de Livraison")
        
        max_distance = st.number_input(
            "Distance maximale (km)",
            value=int(model_settings.get('max_distance_km', 3000)),
            step=100
        )
        default_delay = st.number_input(
            "Délai par défaut (jours)",
            value=int(model_settings.get('default_delay_days', 10)),
            step=1
        )
        
        if st.button("💾 Sauvegarder Configuration", width='stretch'):
            app_config['model_settings']['shipping'] = {
                'max_distance_km': int(max_distance),
                'default_delay_days': int(default_delay)
            }
            save_app_config(app_config)
            st.success("✅ Configuration sauvegardée")
    
    elif selected_model == 'sentiment':
        st.markdown("##### 💬 Paramètres d'Analyse de Sentiment")
        
        threshold_positive = st.slider(
            "Seuil sentiment positif",
            0.0, 1.0,
            float(model_settings.get('threshold_positive', 0.7))
        )
        threshold_negative = st.slider(
            "Seuil sentiment négatif",
            0.0, 1.0,
            float(model_settings.get('threshold_negative', 0.3))
        )
        
        if st.button("💾 Sauvegarder Configuration", width='stretch'):
            app_config['model_settings']['sentiment'] = {
                'threshold_positive': float(threshold_positive),
                'threshold_negative': float(threshold_negative)
            }
            save_app_config(app_config)
            st.success("✅ Configuration sauvegardée")
    
    elif selected_model == 'orders':
        st.markdown("##### 📦 Paramètres de Prédiction de Commandes")
        
        forecast_horizon = st.number_input(
            "Horizon de prévision (mois)",
            value=int(model_settings.get('forecast_horizon', 3)),
            step=1
        )
        confidence_level = st.slider(
            "Niveau de confiance",
            0.0, 1.0,
            float(model_settings.get('confidence_level', 0.95))
        )
        
        if st.button("💾 Sauvegarder Configuration", width='stretch'):
            app_config['model_settings']['orders'] = {
                'forecast_horizon': int(forecast_horizon),
                'confidence_level': float(confidence_level)
            }
            save_app_config(app_config)
            st.success("✅ Configuration sauvegardée")
    
    elif selected_model == 'clustering':
        st.markdown("##### 🎯 Paramètres de Recommandation")
        
        n_recommendations = st.number_input(
            "Nombre de recommandations",
            value=int(model_settings.get('n_recommendations', 4)),
            step=1
        )
        similarity_threshold = st.slider(
            "Seuil de similarité",
            0.0, 1.0,
            float(model_settings.get('similarity_threshold', 0.5))
        )
        
        if st.button("💾 Sauvegarder Configuration", width='stretch'):
            app_config['model_settings']['clustering'] = {
                'n_recommendations': int(n_recommendations),
                'similarity_threshold': float(similarity_threshold)
            }
            save_app_config(app_config)
            st.success("✅ Configuration sauvegardée")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>⚙️ Gestionnaire de Modèles ML</p>
    <p>💡 Astuce: Gardez toujours une copie locale de vos modèles avant de les uploader</p>
</div>
""", unsafe_allow_html=True)
