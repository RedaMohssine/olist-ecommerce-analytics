"""
Gestionnaire de modèles ML avec upload et versioning
"""

import streamlit as st
import joblib
import json
from pathlib import Path
from datetime import datetime
import shutil
import os

MODELS_PATH = Path(__file__).parent.parent / "models"

class ModelManager:
    def __init__(self, model_type):
        """
        Initialise le gestionnaire pour un type de modèle
        
        Args:
            model_type: 'shipping', 'sentiment', 'orders', ou 'clustering'
        """
        self.model_type = model_type
        self.model_dir = MODELS_PATH / model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_file = self.model_dir / "model.joblib"
        self.config_file = self.model_dir / "config.json"
        self.history_dir = self.model_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
    
    def load_model(self):
        """Charge le modèle actif (uploaded) ou le modèle par défaut"""
        try:
            # Priorité 1: Modèle uploadé par l'utilisateur
            if self.model_file.exists():
                return joblib.load(self.model_file)

            # Nouveaux chemins par type (modèles intégrés)
            if self.model_type == 'orders':
                orders_dir = MODELS_PATH / "orders_forecast"
                orders_model = orders_dir / "xgboost_model.pkl"
                if orders_model.exists():
                    return joblib.load(orders_model)

            if self.model_type == 'shipping':
                shipping_dir = MODELS_PATH / "shipping_forecast"
                pipeline_path = shipping_dir / "xgboost_pipeline.pkl"
                if pipeline_path.exists():
                    return joblib.load(pipeline_path)

            if self.model_type == 'clustering':
                rec_dir = MODELS_PATH / "recommendation"
                knn_path = rec_dir / "knn_model.pkl"
                if knn_path.exists():
                    return joblib.load(knn_path)
            
            # Priorité 2: Modèle par défaut pour le sentiment
            if self.model_type == 'sentiment':
                default_model_path = self.model_dir / "sentiment_model.pkl"
                if default_model_path.exists():
                    return joblib.load(default_model_path)
            
            # Priorité 3: Modèle par défaut pour le shipping (XGBoost ou CatBoost)
            if self.model_type == 'shipping':
                # Essayer XGBoost en premier
                xgb_model_path = self.model_dir / "xgboost_shipping_model.json"
                if xgb_model_path.exists():
                    from xgboost import XGBRegressor
                    model = XGBRegressor()
                    model.load_model(str(xgb_model_path))
                    return model
                
                # Fallback sur CatBoost si disponible
                catboost_model_path = self.model_dir / "catboost_shipping_model.cbm"
                if catboost_model_path.exists():
                    try:
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor()
                        model.load_model(str(catboost_model_path))
                        return model
                    except ImportError:
                        pass  # CatBoost not installed
            
            return None
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle {self.model_type}: {e}")
            return None
    
    def save_model(self, model, metadata=None):
        """
        Sauvegarde un nouveau modèle
        
        Args:
            model: Le modèle à sauvegarder
            metadata: Dictionnaire avec infos (accuracy, author, etc.)
        """
        try:
            # Archiver l'ancien modèle si existant
            if self.model_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_file = self.history_dir / f"model_{timestamp}.joblib"
                shutil.copy(self.model_file, archive_file)
            
            # Sauvegarder le nouveau modèle
            joblib.dump(model, self.model_file)
            
            # Sauvegarder les métadonnées
            if metadata is None:
                metadata = {}
            
            metadata['upload_date'] = datetime.now().isoformat()
            metadata['model_type'] = self.model_type
            
            with open(self.config_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return True
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la sauvegarde du modèle: {e}")
            return False
    
    def get_metadata(self):
        """Récupère les métadonnées du modèle actif"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            # Fallback vers config des nouveaux modèles
            if self.model_type == 'orders':
                config_path = MODELS_PATH / "orders_forecast" / "config.json"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            if self.model_type == 'shipping':
                config_path = MODELS_PATH / "shipping_forecast" / "config.json"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            if self.model_type == 'clustering':
                config_path = MODELS_PATH / "recommendation" / "config.json"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            else:
                return {}
        except:
            return {}
    
    def get_history(self):
        """Liste l'historique des modèles archivés"""
        history_files = list(self.history_dir.glob("model_*.joblib"))
        history_files.sort(reverse=True)
        
        return [f.name for f in history_files]
    
    def restore_from_history(self, history_filename):
        """Restaure un modèle depuis l'historique"""
        try:
            history_file = self.history_dir / history_filename
            
            if history_file.exists():
                # Archiver le modèle actuel
                if self.model_file.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = self.history_dir / f"model_backup_{timestamp}.joblib"
                    shutil.copy(self.model_file, backup_file)
                
                # Restaurer
                shutil.copy(history_file, self.model_file)
                
                # Mettre à jour les métadonnées
                metadata = self.get_metadata()
                metadata['restored_from'] = history_filename
                metadata['restore_date'] = datetime.now().isoformat()
                
                with open(self.config_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                return True
        except Exception as e:
            st.error(f"❌ Erreur lors de la restauration: {e}")
            return False
    
    def delete_from_history(self, history_filename):
        """Supprime un modèle de l'historique"""
        try:
            history_file = self.history_dir / history_filename
            if history_file.exists():
                os.remove(history_file)
                return True
        except Exception as e:
            st.error(f"❌ Erreur lors de la suppression: {e}")
            return False
    
    @staticmethod
    def get_all_models_status():
        """Retourne le statut de tous les modèles"""
        model_types = ['shipping', 'sentiment', 'orders', 'clustering']
        status = {}
        
        for model_type in model_types:
            manager = ModelManager(model_type)
            model = manager.load_model()
            metadata = manager.get_metadata()
            
            status[model_type] = {
                'loaded': model is not None,
                'metadata': metadata,
                'history_count': len(manager.get_history())
            }
        
        return status

# Fonctions utilitaires pour chaque modèle spécifique

@st.cache_resource
def load_shipping_model():
    """Charge le modèle de prédiction de livraison"""
    manager = ModelManager('shipping')
    return manager.load_model()

@st.cache_resource
def load_sentiment_model():
    """Charge le modèle de sentiment et son vectoriseur"""
    manager = ModelManager('sentiment')
    model = manager.load_model()
    
    # Charger aussi le vectoriseur TF-IDF
    vectorizer = None
    try:
        vectorizer_path = manager.model_dir / "tfidf_vectorizer.pkl"
        if vectorizer_path.exists():
            vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        st.warning(f"⚠️ Vectoriseur non trouvé: {e}")
    
    return model, vectorizer
    """Charge le modèle d'analyse de sentiment"""
    manager = ModelManager('sentiment')
    return manager.load_model()

@st.cache_resource
def load_orders_model():
    """Charge le modèle de prédiction de commandes"""
    manager = ModelManager('orders')
    return manager.load_model()

@st.cache_resource
def load_clustering_model():
    """Charge le modèle de clustering/recommandation"""
    manager = ModelManager('clustering')
    return manager.load_model()
