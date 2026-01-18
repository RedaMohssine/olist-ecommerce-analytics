"""
🛒 Page Catalogue Produits - Interface E-commerce Client
"""

import streamlit as st
from components.style import load_global_styles
from components.topbar import render_topbar
from components.sidebar import render_sidebar
import pandas as pd
from components.auth import require_auth
from components.translations import get_text
from components.product_card import (
    render_product_card,
    render_product_detail,
    get_cart_count
)
from utils.recommendation_engine import get_recommendation_engine
from utils.data_loader import get_products_with_stats

# Vérification authentification
require_auth()

st.set_page_config(page_title="Catalogue Produits", page_icon="🛒", layout="wide")

# CSS global
load_global_styles()

# Topbar login/logout
render_topbar(key_prefix="catalog_")

# Sidebar menu
render_sidebar()

# Initialisation
if 'show_product_detail' not in st.session_state:
    st.session_state.show_product_detail = False
if 'selected_product_id' not in st.session_state:
    st.session_state.selected_product_id = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# Chargement des données
@st.cache_resource
def load_data():
    products = get_products_with_stats()
    rec_engine = get_recommendation_engine()
    return products, rec_engine

with st.spinner("🔄 Chargement du catalogue..."):
    products_df, recommendation_engine = load_data()

# ========================================
# AFFICHAGE DÉTAIL PRODUIT
# ========================================
if st.session_state.show_product_detail and st.session_state.selected_product_id:
    product = products_df[
        products_df['product_id'] == st.session_state.selected_product_id
    ]
    
    if not product.empty:
        render_product_detail(product.iloc[0], recommendation_engine)
        st.stop()

# ========================================
# HEADER
# ========================================
st.markdown("""
<div class='client-header'>
    <h1 style='margin: 0;'>🛒 Catalogue Produits</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
        Découvrez plus de 32,000 produits avec livraison dans tout le Brésil
    </p>
</div>
""", unsafe_allow_html=True)

# Panier indicator
cart_count = get_cart_count()
if cart_count > 0:
    st.info(f"🛒 **{cart_count} produit(s) dans votre panier**")

st.markdown("---")

# ========================================
# BARRE DE RECHERCHE ET FILTRES
# ========================================
st.markdown("### 🔍 Recherche et Filtres")

# Barre de recherche
search_query = st.text_input(
    "Recherche",
    placeholder=get_text('search'),
    key='search_input',
    label_visibility='collapsed'
)

# Filtres en colonnes
col_filter1, col_filter2, col_filter3, col_filter4 = st.columns([2, 2, 2, 2])

with col_filter1:
    # Filtre catégorie
    categories = ['all'] + sorted(products_df['product_category_name_english'].dropna().unique().tolist())
    selected_category = st.selectbox(
        get_text('category'),
        options=categories,
        format_func=lambda x: get_text('all_categories') if x == 'all' else x.replace('_', ' ').title(),
        key='category_filter'
    )

with col_filter2:
    # Filtre prix
    min_price = float(products_df['price'].min())
    max_price = float(products_df['price'].max())
    
    price_range = st.slider(
        get_text('price'),
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        key='price_filter'
    )

with col_filter3:
    # Filtre note
    min_rating = st.slider(
        get_text('rating'),
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        key='rating_filter'
    )

with col_filter4:
    # Tri
    sort_options = {
        'relevance': get_text('relevance'),
        'price_asc': get_text('price_asc'),
        'price_desc': get_text('price_desc'),
        'rating': get_text('rating_desc')
    }
    
    sort_by = st.selectbox(
        get_text('sort_by'),
        options=list(sort_options.keys()),
        format_func=lambda x: sort_options[x],
        key='sort_filter'
    )

# Filtres additionnels dans un expander
with st.expander("🔧 Filtres Avancés"):
    col1, col2 = st.columns(2)
    
    with col1:
        show_in_stock_only = st.checkbox(
            "En stock uniquement",
            value=True,
            key='stock_filter'
        )
        show_bestsellers_only = st.checkbox(
            "Best-sellers uniquement",
            value=False,
            key='bestseller_filter'
        )
    
    with col2:
        min_reviews = st.number_input(
            "Nombre minimum d'avis",
            min_value=0,
            max_value=100,
            value=0,
            key='reviews_filter'
        )
        
        weight_range = st.slider(
            "Poids (g)",
            min_value=int(products_df['product_weight_g'].min()),
            max_value=int(products_df['product_weight_g'].max()),
            value=(
                int(products_df['product_weight_g'].min()),
                int(products_df['product_weight_g'].max())
            ),
            key='weight_filter'
        )

st.markdown("---")

# ========================================
# APPLICATION DES FILTRES
# ========================================
filtered_products = recommendation_engine.search_products(
    query=search_query,
    category=selected_category if selected_category != 'all' else None,
    price_range=price_range,
    min_rating=min_rating,
    sort_by=sort_by
)

# Compléter les champs manquants depuis le catalogue complet
if 'in_stock' not in filtered_products.columns:
    catalog_fields = [
        'product_id',
        'in_stock',
        'total_sales',
        'review_count',
        'avg_rating',
        'price',
        'product_weight_g',
        'product_length_cm',
        'product_width_cm',
        'product_height_cm',
        'product_category_name_english',
        'product_category_name',
        'freight_value'
    ]
    available_fields = [c for c in catalog_fields if c in products_df.columns]
    if available_fields:
        filtered_products = filtered_products.merge(
            products_df[available_fields],
            on='product_id',
            how='left'
        )
    filtered_products['in_stock'] = filtered_products.get('in_stock', True).fillna(True)
    filtered_products['total_sales'] = filtered_products.get('total_sales', 0).fillna(0)
    filtered_products['review_count'] = filtered_products.get('review_count', 0).fillna(0)
    filtered_products['avg_rating'] = filtered_products.get('avg_rating', 3.0).fillna(3.0)

# Filtres additionnels
if show_in_stock_only:
    filtered_products = filtered_products[filtered_products['in_stock'] == True]

if show_bestsellers_only:
    filtered_products = filtered_products[filtered_products['total_sales'] > 100]

if min_reviews > 0:
    filtered_products = filtered_products[filtered_products['review_count'] >= min_reviews]

if 'product_weight_g' not in filtered_products.columns:
    if 'product_weight_g' in products_df.columns and 'product_id' in filtered_products.columns:
        filtered_products = filtered_products.merge(
            products_df[['product_id', 'product_weight_g']],
            on='product_id',
            how='left'
        )
    filtered_products['product_weight_g'] = filtered_products.get('product_weight_g', 0).fillna(0)

filtered_products = filtered_products[
    (filtered_products['product_weight_g'] >= weight_range[0]) &
    (filtered_products['product_weight_g'] <= weight_range[1])
]

# ========================================
# PAGINATION
# ========================================
PRODUCTS_PER_PAGE = 12
total_products = len(filtered_products)
total_pages = (total_products + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE

# Affichage du nombre de résultats
st.markdown(f"### 📦 {total_products} {get_text('products_found')}")

if total_products == 0:
    st.warning("😕 Aucun produit ne correspond à vos critères. Essayez d'élargir les filtres.")
    st.stop()

# Contrôles de pagination
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    if st.session_state.current_page > 1:
        if st.button("⬅️ Précédent", key='prev_page'):
            st.session_state.current_page -= 1
            st.rerun()

with col_center:
    st.markdown(f"<div style='text-align: center; padding: 0.5rem;'>Page {st.session_state.current_page} / {total_pages}</div>", 
                unsafe_allow_html=True)

with col_right:
    if st.session_state.current_page < total_pages:
        if st.button("Suivant ➡️", key='next_page'):
            st.session_state.current_page += 1
            st.rerun()

# ========================================
# AFFICHAGE DES PRODUITS (GRILLE 3x4)
# ========================================
start_idx = (st.session_state.current_page - 1) * PRODUCTS_PER_PAGE
end_idx = start_idx + PRODUCTS_PER_PAGE
current_products = filtered_products.iloc[start_idx:end_idx]

# Affichage en grille de 3 colonnes
for row in range(0, len(current_products), 3):
    cols = st.columns(3)
    
    for col_idx, col in enumerate(cols):
        idx = row + col_idx
        if idx < len(current_products):
            product = current_products.iloc[idx]
            
            with col:
                render_product_card(product, show_actions=True, key_prefix=f"grid_{start_idx + idx}_")

st.markdown("---")

# Contrôles de pagination (bas de page)
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    if st.session_state.current_page > 1:
        if st.button("⬅️ Précédent", key='prev_page_bottom'):
            st.session_state.current_page -= 1
            st.rerun()

with col_center:
    # Sélection directe de page
    page_options = list(range(1, total_pages + 1))
    selected_page = st.selectbox(
        "Aller à la page",
        options=page_options,
        index=st.session_state.current_page - 1,
        key='page_selector'
    )
    
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

with col_right:
    if st.session_state.current_page < total_pages:
        if st.button("Suivant ➡️", key='next_page_bottom'):
            st.session_state.current_page += 1
            st.rerun()

# ========================================
# SECTION RECOMMANDATIONS GÉNÉRALES
# ========================================
st.markdown("---")
st.markdown("## ⭐ Produits Populaires")

# Récupérer les best-sellers
bestsellers = products_df[
    (products_df['in_stock'] == True) &
    (products_df['total_sales'] > 50)
].nlargest(8, 'total_sales')

if not bestsellers.empty:
    cols = st.columns(4)
    for idx, (_, product) in enumerate(bestsellers.head(4).iterrows()):
        with cols[idx]:
            render_product_card(product, show_actions=True, key_prefix=f"popular_{idx}_")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🛍️ <strong>Olist Marketplace</strong></p>
    <p>🚚 Livraison dans tout le Brésil | 📦 Retours gratuits sous 30 jours | ⭐ Satisfaction garantie</p>
    <p>🔒 Paiement sécurisé | 💳 Plusieurs modes de paiement | 📞 Support client 24/7</p>
</div>
""", unsafe_allow_html=True)
