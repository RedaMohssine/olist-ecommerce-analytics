"""
Moteur de recommandation basé sur KNN (K-Nearest Neighbors)
Utilise le modèle entraîné dans Knn_clustring.ipynb
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
import streamlit as st

class RecommendationEngine:
    """Moteur de recommandation utilisant KNN avec similarité cosinus"""
    
    def __init__(self):
        self.model_dir = Path("models/recommendation")
        self.knn_model = None
        self.preprocessor = None
        self.features_matrix = None
        self.products_df = None
        self.features_info = None
        self.config = None
        
        # Charger le modèle
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle KNN et tous les fichiers associés"""
        try:
            # 1. Charger le modèle KNN
            knn_path = self.model_dir / "knn_model.pkl"
            if knn_path.exists():
                with open(knn_path, 'rb') as f:
                    self.knn_model = pickle.load(f)
            
            # 2. Charger le preprocessor
            preprocessor_path = self.model_dir / "preprocessor.pkl"
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
            
            # 3. Charger la matrice des features
            features_path = self.model_dir / "features_matrix.pkl"
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    self.features_matrix = pickle.load(f)
            
            # 4. Charger le DataFrame des produits
            products_path = self.model_dir / "products_data.pkl"
            if products_path.exists():
                self.products_df = pd.read_pickle(products_path)
            
            # 5. Charger les infos sur les features
            features_info_path = self.model_dir / "features_info.json"
            if features_info_path.exists():
                with open(features_info_path, 'r') as f:
                    self.features_info = json.load(f)
            
            # 6. Charger la config
            config_path = self.model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            
            # Vérifier que tout est chargé
            if all([
                self.knn_model is not None,
                self.preprocessor is not None,
                self.features_matrix is not None,
                self.products_df is not None,
                self.features_info is not None
            ]):
                print("✅ Modèle de recommandation KNN chargé avec succès")
                print(f"   • {len(self.products_df)} produits")
                print(f"   • {self.features_matrix.shape[1]} features")
                print(f"   • K = {self.features_info.get('k_neighbors', 5)}")
            else:
                print("⚠️ Modèle de recommandation incomplet, utilisation de la méthode baseline")
                self._use_baseline()
        
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            self._use_baseline()
    
    def _use_baseline(self):
        """Fallback vers une méthode baseline simple si le modèle n'est pas disponible"""
        from utils.data_loader import get_products_with_stats
        self.products_df = get_products_with_stats()
        self.knn_model = None
    
    def is_model_loaded(self):
        """Vérifie si le modèle KNN est chargé"""
        return self.knn_model is not None
    
    def get_recommendations(self, product_id, n_recommendations=5, similarity_threshold=0.0):
        """
        Retourne des produits similaires basés sur KNN
        
        Args:
            product_id: ID du produit de référence
            n_recommendations: Nombre de recommandations à retourner
            similarity_threshold: Seuil de similarité minimum (0 à 1)
        
        Returns:
            DataFrame avec les produits recommandés et leurs scores de similarité
        """
        
        if not self.is_model_loaded():
            return self._baseline_recommendations(product_id, n_recommendations)
        
        try:
            # Trouver l'index du produit dans le DataFrame
            product_idx = self.products_df[
                self.products_df['product_id'] == product_id
            ].index
            
            if len(product_idx) == 0:
                print(f"⚠️ Produit {product_id} non trouvé")
                return self._get_top_rated(n_recommendations)
            
            product_idx = product_idx[0]
            
            # Obtenir les k+1 voisins les plus proches (inclut le produit lui-même)
            distances, indices = self.knn_model.kneighbors(
                self.features_matrix[product_idx].reshape(1, -1),
                n_neighbors=n_recommendations + 1
            )
            
            # Exclure le produit lui-même (premier résultat)
            neighbor_indices = indices[0][1:]
            neighbor_distances = distances[0][1:]
            
            # Convertir distances cosinus en similarités
            similarities = 1 - neighbor_distances
            
            # Filtrer par seuil de similarité
            valid_mask = similarities >= similarity_threshold
            neighbor_indices = neighbor_indices[valid_mask]
            similarities = similarities[valid_mask]
            
            # Récupérer les produits recommandés
            recommendations = self.products_df.iloc[neighbor_indices].copy()
            recommendations['similarity_score'] = similarities
            recommendations['rank'] = range(1, len(recommendations) + 1)
            
            return recommendations
        
        except Exception as e:
            print(f"❌ Erreur lors de la recommandation: {e}")
            return self._baseline_recommendations(product_id, n_recommendations)
    
    
    def get_recommendations_by_features(self, avg_price, review_score, category, 
                                       volume=None, n_recommendations=5):
        """
        Recommande des produits basés sur des features spécifiques
        
        Args:
            avg_price: Prix moyen souhaité
            review_score: Note souhaitée
            category: Catégorie du produit
            volume: Volume du produit (optionnel)
            n_recommendations: Nombre de recommandations
        
        Returns:
            DataFrame avec les produits recommandés
        """
        
        if not self.is_model_loaded():
            return self._get_top_rated(n_recommendations)
        
        try:
            # Créer un DataFrame avec les features
            features_dict = {
                'avg_price': avg_price,
                'review_score': review_score,
                'product_category_name': category,
                'volume': volume if volume else 10000  # Valeur par défaut
            }
            
            features_df = pd.DataFrame([features_dict])
            
            # Transformer avec le preprocessor
            features_transformed = self.preprocessor.transform(features_df)
            
            # Trouver les voisins
            distances, indices = self.knn_model.kneighbors(
                features_transformed,
                n_neighbors=n_recommendations
            )
            
            # Récupérer les produits
            neighbor_indices = indices[0]
            neighbor_distances = distances[0]
            similarities = 1 - neighbor_distances
            
            recommendations = self.products_df.iloc[neighbor_indices].copy()
            recommendations['similarity_score'] = similarities
            recommendations['rank'] = range(1, len(recommendations) + 1)
            
            return recommendations
        
        except Exception as e:
            print(f"❌ Erreur lors de la recommandation par features: {e}")
            return self._get_top_rated(n_recommendations)
    
    def search_products(self, query, category=None, price_range=None, 
                       min_rating=None, sort_by='relevance'):
        """
        Recherche de produits avec filtres
        
        Args:
            query: Terme de recherche
            category: Filtre par catégorie
            price_range: Tuple (min, max)
            min_rating: Note minimale
            sort_by: 'relevance', 'price_asc', 'price_desc', 'rating', 'popularity'
        
        Returns:
            DataFrame filtré
        """
        if self.products_df is None:
            return pd.DataFrame()
        
        results = self.products_df.copy()
        
        # Filtre par recherche textuelle
        if query:
            query_lower = query.lower()
            results = results[
                results['product_category_name'].str.lower().str.contains(query_lower, na=False)
            ]
        
        # Filtre par catégorie
        if category and category != 'Toutes':
            results = results[results['product_category_name'] == category]
        
        # Filtre par prix
        if price_range:
            min_price, max_price = price_range
            results = results[
                (results['avg_price'] >= min_price) &
                (results['avg_price'] <= max_price)
            ]
        
        # Filtre par note
        if min_rating:
            results = results[results['review_score'] >= min_rating]
        
        # Tri
        if sort_by == 'price_asc':
            results = results.sort_values('avg_price', ascending=True)
        elif sort_by == 'price_desc':
            results = results.sort_values('avg_price', ascending=False)
        elif sort_by == 'rating':
            results = results.sort_values('review_score', ascending=False)
        elif sort_by == 'popularity':
            results = results.sort_values('order_count', ascending=False)
        elif sort_by == 'relevance':
            # Score de pertinence combiné
            results['relevance_score'] = (
                results['review_score'] * 0.4 +
                np.log1p(results['order_count']) * 0.6
            )
            results = results.sort_values('relevance_score', ascending=False)
        
        return results
    
    def get_categories(self):
        """Retourne la liste des catégories disponibles"""
        if self.products_df is not None:
            return sorted(self.products_df['product_category_name'].unique().tolist())
        return []
    
    def get_category_stats(self):
        """Retourne des statistiques par catégorie"""
        if self.products_df is None:
            return pd.DataFrame()
        
        stats = self.products_df.groupby('product_category_name').agg({
            'product_id': 'count',
            'avg_price': 'mean',
            'review_score': 'mean',
            'order_count': 'sum'
        }).reset_index()
        
        stats.columns = ['category', 'n_products', 'avg_price', 'avg_rating', 'total_orders']
        stats = stats.sort_values('total_orders', ascending=False)
        
        return stats
    
    def get_model_info(self):
        """Retourne les informations sur le modèle"""
        if self.config:
            return self.config
        return {
            "model_name": "Baseline Recommendations",
            "model_type": "simple",
            "status": "Modèle KNN non chargé"
        }
    
    # ========================================
    # MÉTHODES BASELINE (FALLBACK)
    # ========================================
    
    def _baseline_recommendations(self, product_id, n_recommendations):
        """Méthode baseline simple basée sur la catégorie et les notes"""
        
        # Récupérer le produit de référence
        product = self.products_df[self.products_df['product_id'] == product_id]
        
        if product.empty:
            return self._get_top_rated(n_recommendations)
        
        product = product.iloc[0]
        
        # Stratégie: Même catégorie + bonnes notes
        same_category = self.products_df[
            (self.products_df['product_category_name'] == product['product_category_name']) &
            (self.products_df['product_id'] != product_id)
        ].copy()
        
        if len(same_category) >= n_recommendations:
            # Trier par note et popularité
            same_category['score'] = (
                same_category['review_score'] * 0.6 + 
                np.log1p(same_category['order_count']) * 0.4
            )
            return same_category.nlargest(n_recommendations, 'score')
        
        # Sinon, retourner les meilleurs produits
        return self._get_top_rated(n_recommendations)
    
    def _get_top_rated(self, n):
        """Retourne les produits les mieux notés"""
        if self.products_df is None:
            return pd.DataFrame()
        
        top_products = self.products_df[
            self.products_df['review_score'] >= 4.0
        ].copy()
        
        if len(top_products) == 0:
            top_products = self.products_df.copy()
        
        top_products['score'] = (
            top_products['review_score'] * 0.5 + 
            np.log1p(top_products['order_count']) * 0.5
        )
        
        return top_products.nlargest(min(n, len(top_products)), 'score')


# ========================================
# FONCTION POUR STREAMLIT
# ========================================

@st.cache_resource
def get_recommendation_engine():
    """Retourne une instance du moteur de recommandation (cached)"""
    return RecommendationEngine()
