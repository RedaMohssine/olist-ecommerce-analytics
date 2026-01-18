"""
Gestionnaire du modèle de prédiction des commandes mensuelles
Utilise le modèle XGBoost entraîné dans Nombre_commande_ParMois V111.ipynb
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st

class OrdersForecastModel:
    """Modèle de prédiction des quantités vendues par produit par mois"""
    
    def __init__(self):
        # Chemin absolu vers le dossier models
        import os
        current_dir = Path(__file__).parent.parent  # streamlit_app/
        self.model_dir = current_dir / "models" / "orders_forecast"
        
        self.model = None
        self.feature_names = None
        self.config = None
        self.best_params = None
        
        # Charger le modèle
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle XGBoost et tous les fichiers associés"""
        try:
            # 1. Charger le modèle XGBoost
            model_path = self.model_dir / "xgboost_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Modèle XGBoost chargé: {model_path}")
            else:
                print(f"❌ Fichier modèle introuvable: {model_path}")
                return
            
            # 2. Charger les noms des features
            features_path = self.model_dir / "feature_names.pkl"
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                print(f"✅ Feature names chargés: {len(self.feature_names)} features")
            else:
                print(f"❌ Fichier features introuvable: {features_path}")
                self.model = None
                return
            
            # 3. Charger la config
            config_path = self.model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"✅ Configuration chargée")
            else:
                print(f"❌ Fichier config introuvable: {config_path}")
                self.model = None
                return
            
            # 4. Charger les paramètres
            params_path = self.model_dir / "best_params.json"
            if params_path.exists():
                with open(params_path, 'r') as f:
                    self.best_params = json.load(f)
                print(f"✅ Paramètres chargés")
            else:
                print(f"⚠️ Fichier params introuvable (optionnel): {params_path}")
            
            # Résumé final
            print("\n" + "="*60)
            print("✅ MODÈLE DE PRÉDICTION DES COMMANDES CHARGÉ")
            print("="*60)
            print(f"• Features: {self.config['data_info']['n_features']}")
            print(f"• MAE: {self.config['metrics']['mae']:.4f}")
            print(f"• R²: {self.config['metrics']['r2_score']:.4f}")
            print("="*60 + "\n")
        
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def is_model_loaded(self):
        """Vérifie si le modèle est chargé"""
        return self.model is not None
    
    def get_model_info(self):
        """Retourne les informations sur le modèle"""
        if self.config:
            return self.config
        return {
            "model_name": "Orders Forecast Model",
            "status": "Modèle non chargé"
        }
    
    def prepare_features(self, product_data, target_month, historical_sales=None):
        """
        Prépare les features pour la prédiction
        
        Args:
            product_data: Dict avec les caractéristiques du produit
            target_month: Date du mois à prédire (datetime)
            historical_sales: Dict optionnel avec l'historique des ventes
        
        Returns:
            DataFrame avec les features préparées
        """
        if not self.is_model_loaded():
            return None
        
        # Créer le dictionnaire de features
        features = {}
        
        # Features temporelles
        features['month'] = target_month.month
        features['year'] = target_month.year
        features['quarter'] = (target_month.month - 1) // 3 + 1
        features['month_sin'] = np.sin(2 * np.pi * target_month.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * target_month.month / 12)
        
        # Saison (hémisphère sud)
        if target_month.month in [12, 1, 2]:
            features['season'] = 0  # été
        elif target_month.month in [3, 4, 5]:
            features['season'] = 1  # automne
        elif target_month.month in [6, 7, 8]:
            features['season'] = 2  # hiver
        else:
            features['season'] = 3  # printemps
        
        # Événements spéciaux
        features['is_black_friday'] = 1 if target_month.month == 11 else 0
        features['is_christmas'] = 1 if target_month.month == 12 else 0
        features['is_end_year'] = 1 if target_month.month in [11, 12] else 0
        
        # Features produit
        features['price'] = product_data.get('price', 100)
        features['freight_value'] = product_data.get('freight_value', 20)
        features['payment_value'] = product_data.get('payment_value', features['price'])
        features['review_score'] = product_data.get('review_score', 4.0)
        features['product_weight_g'] = product_data.get('weight_g', 1000)
        features['product_volume_cm3'] = product_data.get('volume_cm3', 10000)
        features['product_density'] = features['product_weight_g'] / (features['product_volume_cm3'] + 1e-6)
        
        # Ratios
        features['price_freight_ratio'] = features['price'] / (features['freight_value'] + 1e-6)
        features['price_per_kg'] = features['price'] / ((features['product_weight_g'] / 1000) + 1e-6)
        
        # Features historiques (lags et rolling)
        if historical_sales:
            # Lags
            for lag in [1, 2, 3, 6, 12]:
                features[f'lag_{lag}'] = historical_sales.get(f'lag_{lag}', 0)
            
            # Rolling means et std
            for window in [3, 6, 12]:
                features[f'rolling_mean_{window}'] = historical_sales.get(f'rolling_mean_{window}', 0)
                features[f'rolling_std_{window}'] = historical_sales.get(f'rolling_std_{window}', 0)
            
            # Autres features historiques
            features['cumulative_sales_past'] = historical_sales.get('cumulative_sales_past', 0)
            features['cv_3m'] = historical_sales.get('cv_3m', 0)
            features['trend_3m'] = historical_sales.get('trend_3m', 0)
            features['category_avg_sales'] = historical_sales.get('category_avg_sales', 1)
            features['category_std_sales'] = historical_sales.get('category_std_sales', 0)
            features['sales_vs_category'] = historical_sales.get('sales_vs_category', 1)
            features['product_age_months'] = historical_sales.get('product_age_months', 1)
            features['cumulative_sales'] = historical_sales.get('cumulative_sales', 0)
            features['lifetime_avg_sales'] = historical_sales.get('lifetime_avg_sales', 0)
            features['had_sales_last_year'] = historical_sales.get('had_sales_last_year', 0)
            features['sales_vs_lifetime_avg'] = historical_sales.get('sales_vs_lifetime_avg', 1)
            features['rolling3_x_trend'] = features['rolling_mean_3'] * features['trend_3m']
        else:
            # Valeurs par défaut si pas d'historique
            for lag in [1, 2, 3, 6, 12]:
                features[f'lag_{lag}'] = 0
            for window in [3, 6, 12]:
                features[f'rolling_mean_{window}'] = 0
                features[f'rolling_std_{window}'] = 0
            features['cumulative_sales_past'] = 0
            features['cv_3m'] = 0
            features['trend_3m'] = 0
            features['category_avg_sales'] = 1
            features['category_std_sales'] = 0
            features['sales_vs_category'] = 1
            features['product_age_months'] = 1
            features['cumulative_sales'] = 0
            features['lifetime_avg_sales'] = 0
            features['had_sales_last_year'] = 0
            features['sales_vs_lifetime_avg'] = 1
            features['rolling3_x_trend'] = 0
        
        # Créer le DataFrame avec les features dans le bon ordre
        df = pd.DataFrame([features])
        
        # S'assurer que toutes les features requises sont présentes
        for feature_name in self.feature_names:
            if feature_name not in df.columns:
                df[feature_name] = 0
        
        # Retourner seulement les features utilisées par le modèle
        return df[self.feature_names]
    
    def predict(self, product_data, target_month, historical_sales=None):
        """
        Prédit la quantité vendue pour un produit donné un mois donné
        
        Args:
            product_data: Dict avec les caractéristiques du produit
            target_month: Date du mois à prédire
            historical_sales: Dict optionnel avec l'historique des ventes
        
        Returns:
            Int: Quantité prédite (arrondie, min=0)
        """
        if not self.is_model_loaded():
            return None
        
        try:
            # Préparer les features
            X = self.prepare_features(product_data, target_month, historical_sales)
            
            if X is None:
                return None
            
            # Prédire
            prediction = self.model.predict(X)[0]
            
            # Arrondir et s'assurer que c'est >= 0
            prediction_int = max(0, int(round(prediction)))
            
            return prediction_int
        
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return None
    
    def predict_multiple_months(self, product_data, start_month, n_months=12, historical_sales=None):
        """
        Prédit les ventes pour plusieurs mois consécutifs
        
        Args:
            product_data: Dict avec les caractéristiques du produit
            start_month: Date du premier mois à prédire
            n_months: Nombre de mois à prédire
            historical_sales: Dict optionnel avec l'historique des ventes
        
        Returns:
            DataFrame avec les prédictions par mois
        """
        if not self.is_model_loaded():
            return None
        
        predictions = []
        current_month = start_month
        
        for i in range(n_months):
            prediction = self.predict(product_data, current_month, historical_sales)
            
            predictions.append({
                'month': current_month,
                'month_str': current_month.strftime('%Y-%m'),
                'predicted_quantity': prediction
            })
            
            # Passer au mois suivant
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.month + 1)
        
        return pd.DataFrame(predictions)
    
    def get_feature_importance(self, top_n=10):
        """Retourne l'importance des features"""
        if not self.is_model_loaded():
            return None
        
        try:
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance.head(top_n)
        except:
            return None


# ========================================
# FONCTION POUR STREAMLIT
# ========================================

@st.cache_resource
def get_orders_forecast_model():
    """Retourne une instance du modèle de prédiction (cached)"""
    return OrdersForecastModel()
