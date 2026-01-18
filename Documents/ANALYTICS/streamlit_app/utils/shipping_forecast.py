"""
Gestionnaire du modèle de prédiction des délais de livraison
Utilise le modèle XGBoost entraîné dans analytics_shipping_timev2.ipynb
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from datetime import datetime
import streamlit as st
from math import radians, sin, cos, sqrt, atan2

class ShippingForecastModel:
    """Modèle de prédiction des délais de livraison"""
    
    def __init__(self):
        # Chemin absolu vers le dossier models
        current_dir = Path(__file__).parent.parent  # streamlit_app/
        self.model_dir = current_dir / "models" / "shipping_forecast"
        
        self.pipeline = None
        self.feature_names = None
        self.seller_avg_dispatch = None
        self.global_avg_dispatch = None
        self.config = None
        
        # Charger le modèle
        self._load_model()
    
    def _load_model(self):
        """Charge le pipeline XGBoost et tous les fichiers associés"""
        try:
            # 1. Charger le pipeline complet
            pipeline_path = self.model_dir / "xgboost_pipeline.pkl"
            if pipeline_path.exists():
                with open(pipeline_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                print(f"✅ Pipeline XGBoost chargé: {pipeline_path}")
            else:
                print(f"❌ Fichier pipeline introuvable: {pipeline_path}")
                return
            
            # 2. Charger les noms des features
            features_path = self.model_dir / "feature_names.pkl"
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                print(f"✅ Feature names chargés: {len(self.feature_names)} features")
            else:
                print(f"❌ Fichier features introuvable: {features_path}")
                self.pipeline = None
                return
            
            # 3. Charger seller_avg_dispatch
            seller_path = self.model_dir / "seller_avg_dispatch.pkl"
            if seller_path.exists():
                with open(seller_path, 'rb') as f:
                    self.seller_avg_dispatch = pickle.load(f)
                print(f"✅ Seller avg dispatch chargé: {len(self.seller_avg_dispatch)} vendeurs")
            else:
                print(f"❌ Fichier seller avg dispatch introuvable: {seller_path}")
                self.pipeline = None
                return
            
            # 4. Charger global_avg_dispatch
            global_path = self.model_dir / "global_avg_dispatch.pkl"
            if global_path.exists():
                with open(global_path, 'rb') as f:
                    self.global_avg_dispatch = pickle.load(f)
                print(f"✅ Global avg dispatch chargé: {self.global_avg_dispatch:.2f}h")
            else:
                print(f"❌ Fichier global avg dispatch introuvable: {global_path}")
                self.pipeline = None
                return
            
            # 5. Charger la config
            config_path = self.model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"✅ Configuration chargée")
            else:
                print(f"⚠️ Fichier config introuvable (optionnel): {config_path}")
            
            # Résumé final
            print("\n" + "="*60)
            print("✅ MODÈLE DE PRÉDICTION DES LIVRAISONS CHARGÉ")
            print("="*60)
            if self.config:
                print(f"• Features: {self.config['data_info']['n_features']}")
                print(f"• MAE: {self.config['metrics']['mae']:.4f} jours")
                print(f"• R²: {self.config['metrics']['r2_score']:.4f}")
            print("="*60 + "\n")
        
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None
    
    def is_model_loaded(self):
        """Vérifie si le modèle est chargé"""
        return self.pipeline is not None
    
    def get_model_info(self):
        """Retourne les informations sur le modèle"""
        if self.config:
            return self.config
        return {
            "model_name": "Shipping Time Forecast Model",
            "status": "Modèle non chargé"
        }
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calcule la distance haversine entre deux points géographiques"""
        R = 6371  # Rayon de la Terre en km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def prepare_features(self, order_data, geolocation_data=None):
        """
        Prépare les features pour la prédiction
        
        Args:
            order_data: Dict avec les caractéristiques de la commande
            geolocation_data: Dict optionnel avec les données de géolocalisation
        
        Returns:
            DataFrame avec les features préparées
        """
        if not self.is_model_loaded():
            return None
        
        # Créer le dictionnaire de features
        features = {}
        
        # Features temporelles
        purchase_date = order_data.get('purchase_date', datetime.now())
        features['purchase_dayofweek'] = purchase_date.weekday()
        features['purchase_month'] = purchase_date.month
        
        # Saison
        month = purchase_date.month
        if month in [12, 1, 2]:
            features['purchase_season'] = 'winter'
        elif month in [3, 4, 5]:
            features['purchase_season'] = 'spring'
        elif month in [6, 7, 8]:
            features['purchase_season'] = 'summer'
        else:
            features['purchase_season'] = 'fall'
        
        # Features de délai (à définir par défaut si non fourni)
        features['time_to_approve_order'] = order_data.get('time_to_approve_order', 2.0)  # 2h par défaut
        
        # Features géographiques
        if geolocation_data:
            customer_lat = geolocation_data.get('customer_lat')
            customer_lng = geolocation_data.get('customer_lng')
            seller_lat = geolocation_data.get('seller_lat')
            seller_lng = geolocation_data.get('seller_lng')
            
            if all([customer_lat, customer_lng, seller_lat, seller_lng]):
                distance = self.haversine_distance(customer_lat, customer_lng, seller_lat, seller_lng)
                features['distance_customer_seller'] = distance
                features['circuity_distance'] = distance * 1.3
            else:
                features['distance_customer_seller'] = 500  # Valeur par défaut
                features['circuity_distance'] = 650
        else:
            features['distance_customer_seller'] = 500
            features['circuity_distance'] = 650
        
        # Features de commande
        features['num_items'] = order_data.get('num_items', 1)
        features['num_unique_sellers'] = order_data.get('num_unique_sellers', 1)
        features['total_freight_value'] = order_data.get('total_freight_value', 20.0)
        features['total_payment_value'] = order_data.get('total_payment_value', 100.0)
        features['num_payments'] = order_data.get('num_payments', 1)
        
        # Features produit
        features['price'] = order_data.get('price', 50.0)
        features['freight_value'] = order_data.get('freight_value', 15.0)
        features['product_weight_g'] = order_data.get('product_weight_g', 500)
        features['product_length_cm'] = order_data.get('product_length_cm', 20)
        features['product_height_cm'] = order_data.get('product_height_cm', 10)
        features['product_width_cm'] = order_data.get('product_width_cm', 15)
        features['product_name_lenght'] = order_data.get('product_name_lenght', 40)
        features['product_description_lenght'] = order_data.get('product_description_lenght', 500)
        
        # Calculer volume
        features['product_volume_cm3'] = (
            features['product_length_cm'] * 
            features['product_height_cm'] * 
            features['product_width_cm']
        )
        
        # Features catégorielles
        features['product_category_name'] = order_data.get('product_category_name', 'None')
        features['customer_state'] = order_data.get('customer_state', 'SP')
        features['seller_state'] = order_data.get('seller_state', 'SP')
        features['customer_state_geo'] = order_data.get('customer_state_geo', 'SP')
        features['seller_state_geo'] = order_data.get('seller_state_geo', 'SP')
        
        # Feature clé: seller_avg_dispatch
        seller_id = order_data.get('seller_id')
        if seller_id and seller_id in self.seller_avg_dispatch.index:
            features['seller_avg_dispatch'] = self.seller_avg_dispatch[seller_id]
        else:
            features['seller_avg_dispatch'] = self.global_avg_dispatch
        
        # Créer le DataFrame
        df = pd.DataFrame([features])
        
        # S'assurer que toutes les features requises sont présentes
        for feature_name in self.feature_names:
            if feature_name not in df.columns:
                df[feature_name] = 0 if df.dtypes.get(feature_name, 'float') != 'object' else 'None'
        
        # Retourner seulement les features utilisées par le modèle, dans le bon ordre
        return df[self.feature_names]
    
    def predict(self, order_data, geolocation_data=None):
        """
        Prédit le délai de livraison pour une commande
        
        Args:
            order_data: Dict avec les caractéristiques de la commande
            geolocation_data: Dict optionnel avec les données de géolocalisation
        
        Returns:
            Float: Délai de livraison prédit en jours
        """
        if not self.is_model_loaded():
            return None
        
        try:
            # Préparer les features
            X = self.prepare_features(order_data, geolocation_data)
            
            if X is None:
                return None
            
            # Prédire
            prediction = self.pipeline.predict(X)[0]
            
            # S'assurer que c'est >= 0
            prediction = max(0, prediction)
            
            return prediction
        
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_batch(self, orders_df):
        """
        Prédit les délais de livraison pour plusieurs commandes
        
        Args:
            orders_df: DataFrame avec les commandes
        
        Returns:
            Array: Délais de livraison prédits
        """
        if not self.is_model_loaded():
            return None
        
        try:
            predictions = []
            
            for idx, row in orders_df.iterrows():
                order_data = row.to_dict()
                prediction = self.predict(order_data)
                predictions.append(prediction)
            
            return np.array(predictions)
        
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction batch: {e}")
            return None


# ========================================
# FONCTION POUR STREAMLIT
# ========================================

@st.cache_resource
def get_shipping_forecast_model():
    """Retourne une instance du modèle de prédiction (cached)"""
    return ShippingForecastModel()
