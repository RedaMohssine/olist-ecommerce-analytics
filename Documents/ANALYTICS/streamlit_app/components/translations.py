"""
Système de traduction multilingue (FR/EN/PT-BR)
"""

import streamlit as st

TRANSLATIONS = {
    'fr': {
        'select_profile': '-- Sélectionnez un profil --',
        'user_profile': 'Profil Utilisateur',
        'logged_as': 'Connecté en tant que',
        'logout': 'Déconnexion',
        'language': 'Langue',
        'search': 'Rechercher des produits...',
        'filters': 'Filtres',
        'category': 'Catégorie',
        'all_categories': 'Toutes les catégories',
        'price': 'Prix',
        'rating': 'Note',
        'sort_by': 'Trier par',
        'relevance': 'Pertinence',
        'price_asc': 'Prix croissant',
        'price_desc': 'Prix décroissant',
        'rating_desc': 'Meilleures notes',
        'products_found': 'produits trouvés',
        'add_to_cart': 'Ajouter au panier',
        'product_details': 'Détails du produit',
        'reviews': 'Avis clients',
        'delivery_estimate': 'Estimation de livraison',
        'days': 'jours',
        'recommendations': 'Vous aimerez aussi',
        'sentiment_analysis': 'Analyse des sentiments',
        'positive': 'Positif',
        'negative': 'Négatif',
        'neutral': 'Neutre',
        'weight': 'Poids',
        'dimensions': 'Dimensions',
        'seller': 'Vendeur',
        'in_stock': 'En stock',
        'out_of_stock': 'Rupture de stock',
    },
    'en': {
        'select_profile': '-- Select a profile --',
        'user_profile': 'User Profile',
        'logged_as': 'Logged in as',
        'logout': 'Logout',
        'language': 'Language',
        'search': 'Search products...',
        'filters': 'Filters',
        'category': 'Category',
        'all_categories': 'All categories',
        'price': 'Price',
        'rating': 'Rating',
        'sort_by': 'Sort by',
        'relevance': 'Relevance',
        'price_asc': 'Price: Low to High',
        'price_desc': 'Price: High to Low',
        'rating_desc': 'Top rated',
        'products_found': 'products found',
        'add_to_cart': 'Add to cart',
        'product_details': 'Product details',
        'reviews': 'Customer reviews',
        'delivery_estimate': 'Delivery estimate',
        'days': 'days',
        'recommendations': 'You might also like',
        'sentiment_analysis': 'Sentiment Analysis',
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral',
        'weight': 'Weight',
        'dimensions': 'Dimensions',
        'seller': 'Seller',
        'in_stock': 'In stock',
        'out_of_stock': 'Out of stock',
    },
    'pt': {
        'select_profile': '-- Selecione um perfil --',
        'user_profile': 'Perfil do Usuário',
        'logged_as': 'Conectado como',
        'logout': 'Sair',
        'language': 'Idioma',
        'search': 'Pesquisar produtos...',
        'filters': 'Filtros',
        'category': 'Categoria',
        'all_categories': 'Todas as categorias',
        'price': 'Preço',
        'rating': 'Avaliação',
        'sort_by': 'Ordenar por',
        'relevance': 'Relevância',
        'price_asc': 'Menor preço',
        'price_desc': 'Maior preço',
        'rating_desc': 'Mais bem avaliados',
        'products_found': 'produtos encontrados',
        'add_to_cart': 'Adicionar ao carrinho',
        'product_details': 'Detalhes do produto',
        'reviews': 'Avaliações',
        'delivery_estimate': 'Prazo de entrega',
        'days': 'dias',
        'recommendations': 'Você também pode gostar',
        'sentiment_analysis': 'Análise de sentimentos',
        'positive': 'Positivo',
        'negative': 'Negativo',
        'neutral': 'Neutro',
        'weight': 'Peso',
        'dimensions': 'Dimensões',
        'seller': 'Vendedor',
        'in_stock': 'Em estoque',
        'out_of_stock': 'Esgotado',
    }
}

def get_text(key, lang=None):
    """Récupère une traduction selon la langue active"""
    if lang is None:
        lang = st.session_state.get('language', 'fr')
    
    return TRANSLATIONS.get(lang, TRANSLATIONS['fr']).get(key, key)

def set_language(lang):
    """Change la langue de l'interface"""
    st.session_state.language = lang

def render_language_selector():
    """Affiche le sélecteur de langue dans la sidebar"""
    with st.sidebar:
        languages = {
            'fr': '🇫🇷 Français',
            'en': '🇬🇧 English',
            'pt': '🇧🇷 Português'
        }
        
        current_lang = st.session_state.get('language', 'fr')
        
        selected_lang = st.selectbox(
            get_text('language'),
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            index=list(languages.keys()).index(current_lang),
            key='language_selector'
        )
        
        if selected_lang != current_lang:
            set_language(selected_lang)
            st.rerun()
