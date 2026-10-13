"""
Composant de carte produit pour l'interface e-commerce
"""

import streamlit as st
import pandas as pd
from utils.data_loader import get_product_review_stats, get_product_review_comments, get_state_options, get_product_seller_state_map, get_products_with_stats
from utils.shipping_forecast import get_shipping_forecast_model
from components.translations import get_text

def _compact_html(html):
    """Retire l'indentation et les lignes vides : sinon Markdown affiche le HTML comme du code"""
    return "\n".join(line.strip() for line in html.splitlines() if line.strip())

def render_product_card(product, show_actions=True, key_prefix=""):
    """
    Affiche une carte produit
    
    Args:
        product: Series pandas avec les infos du produit
        show_actions: Afficher les boutons d'action
        key_prefix: Préfixe unique pour éviter les collisions de clés
    """
    
    # Image placeholder
    product_emoji = get_product_emoji(product.get('product_category_name_english', 'other'))
    
    # Badge de stock
    stock_badge = "🟢 " + get_text('in_stock') if product.get('in_stock', True) else "🔴 " + get_text('out_of_stock')
    stock_class = "badge-in-stock" if product.get('in_stock', True) else "badge-out-of-stock"
    
    # Badge bestseller si beaucoup de ventes
    is_bestseller = product.get('total_sales', 0) > 100
    
    # HTML de la carte
    st.markdown(_compact_html(f"""
    <div class="product-card fade-in">
        <div class="product-image">
            {product_emoji}
        </div>
        <div class="product-info">
            <div class="product-badge {stock_class}">
                {stock_badge}
            </div>
            {f'<div class="product-badge badge-bestseller">⭐ Best-seller</div>' if is_bestseller else ''}
            
            <h3 class="product-title">
                {product.get('product_category_name_english', 'Product').title()}
            </h3>
            
            <div class="product-rating">
                {'⭐' * int(product.get('avg_rating', 0))}
                <span style="color: #666; font-size: 0.9rem;">
                    {product.get('avg_rating', 0):.1f} ({int(product.get('review_count', 0))} avis)
                </span>
            </div>
            
            <div class="product-price">
                R$ {product.get('price', 0):.2f}
            </div>
            
            <div style="font-size: 0.85rem; color: #666; margin: 0.5rem 0;">
                📦 {product.get('product_weight_g', 0):.0f}g
                {' | 📏 ' + str(int(product.get('product_length_cm', 0))) + 'x' + str(int(product.get('product_width_cm', 0))) + 'x' + str(int(product.get('product_height_cm', 0))) + 'cm' if not pd.isna(product.get('product_length_cm')) else ''}
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)
    
    # Boutons d'action
    if show_actions and product.get('in_stock', True):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👁️ Voir", key=f"{key_prefix}view_{product['product_id']}", width='stretch'):
                st.session_state.selected_product_id = product['product_id']
                st.session_state.show_product_detail = True
                st.rerun()
        
        with col2:
            if st.button("🛒 Ajouter", key=f"{key_prefix}cart_{product['product_id']}", width='stretch'):
                add_to_cart(product['product_id'])
                st.success("✅ Produit ajouté !")

def render_product_detail(product, recommendation_engine):
    """
    Affiche la page détaillée d'un produit
    
    Args:
        product: Series pandas avec les infos du produit
        recommendation_engine: Instance du moteur de recommandation
    """
    
    # Retour au catalogue
    if st.button("← Retour au catalogue"):
        st.session_state.show_product_detail = False
        st.rerun()
    
    st.markdown("---")
    
    # Layout en 2 colonnes
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Image grande
        product_emoji = get_product_emoji(product.get('product_category_name_english', 'other'))
        
        st.markdown(f"""
        <div class="product-detail-image">
            {product_emoji}
        </div>
        """, unsafe_allow_html=True)
        
        # Galerie d'images (placeholder)
        st.markdown("##### 📸 Galerie")
        cols = st.columns(4)
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"""
                <div style="width: 100%; height: 80px; background: linear-gradient(135deg, #FEDD00 0%, #009739 100%); 
                     border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                    {product_emoji}
                </div>
                """, unsafe_allow_html=True)
    
    with col_right:
        # Titre et catégorie
        st.markdown(f"### {product.get('product_category_name_english', 'Product').title()}")
        st.markdown(f"**Catégorie:** {product.get('product_category_name', 'N/A')}")
        
        # Note
        rating = product.get('avg_rating', 0)
        review_count = int(product.get('review_count', 0))
        st.markdown(f"{'⭐' * int(rating)} **{rating:.1f}** ({review_count} avis)")
        
        # Prix
        st.markdown(f"## 💰 R$ {product.get('price', 0):.2f}")
        
        # Statut stock
        if product.get('in_stock', True):
            st.success("✅ " + get_text('in_stock'))
        else:
            st.error("❌ " + get_text('out_of_stock'))
        
        # Caractéristiques
        st.markdown("---")
        st.markdown("#### 📋 Caractéristiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(get_text('weight'), f"{product.get('product_weight_g', 0):.0f} g")
            st.metric("Volume", f"{product.get('product_volume_cm3', 0):.0f} cm³" if 'product_volume_cm3' in product else "N/A")
        
        with col2:
            dimensions = f"{int(product.get('product_length_cm', 0))} x {int(product.get('product_width_cm', 0))} x {int(product.get('product_height_cm', 0))} cm"
            st.metric(get_text('dimensions'), dimensions)
            st.metric("Frais de port", f"R$ {product.get('freight_value', 0):.2f}")
        
        # Estimation de livraison (modèle XGBoost)
        st.markdown("---")
        st.markdown(f"#### 🚚 {get_text('delivery_estimate')}")

        state_options = get_state_options() or ['SP', 'RJ', 'MG']
        customer_state = st.selectbox(
            "État du client",
            options=state_options,
            key=f"cust_state_{product.get('product_id')}"
        )

        seller_state_map = get_product_seller_state_map()
        seller_state = seller_state_map.get(product.get('product_id')) or customer_state

        if st.button("📦 Calculer la livraison", key=f"ship_calc_{product.get('product_id')}"):
            shipping_model = get_shipping_forecast_model()
            state_coords = {
                'SP': (-23.55, -46.63),
                'RJ': (-22.91, -43.17),
                'MG': (-19.92, -43.94),
                'PR': (-25.42, -49.27),
                'RS': (-30.03, -51.23),
                'BA': (-12.97, -38.51),
                'SC': (-27.59, -48.55)
            }

            customer_lat, customer_lng = state_coords.get(customer_state, (-23.55, -46.63))
            seller_lat, seller_lng = state_coords.get(seller_state, (-23.55, -46.63))

            order_data = {
                'seller_id': None,
                'customer_state': customer_state,
                'seller_state': seller_state,
                'customer_zip_code_prefix': 10000,
                'seller_zip_code_prefix': 10000,
                'product_weight_g': product.get('product_weight_g', 0),
                'price': product.get('price', 0),
                'product_length_cm': product.get('product_length_cm', 0),
                'product_width_cm': product.get('product_width_cm', 0),
                'product_height_cm': product.get('product_height_cm', 0),
                'freight_value': product.get('freight_value', 0),
                'num_items': 1,
                'num_unique_sellers': 1,
                'total_freight_value': product.get('freight_value', 0),
                'total_payment_value': product.get('price', 0),
                'num_payments': 1,
                'purchase_date': pd.Timestamp.now(),
                'product_category_name': product.get('product_category_name', 'None'),
                'customer_state_geo': customer_state,
                'seller_state_geo': seller_state,
                'product_name_lenght': product.get('product_name_lenght', 50),
                'product_description_lenght': product.get('product_description_lenght', 500)
            }

            geolocation_data = {
                'customer_lat': customer_lat,
                'customer_lng': customer_lng,
                'seller_lat': seller_lat,
                'seller_lng': seller_lng
            }

            prediction_days = shipping_model.predict(order_data, geolocation_data)
            if prediction_days is not None:
                st.info(f"📦 Livraison estimée: **{prediction_days:.1f} {get_text('days')}**")
            else:
                st.warning("Impossible d'estimer la livraison pour ce produit.")
        
        # Boutons d'action
        st.markdown("---")
        
        if product.get('in_stock', True):
            col1, col2 = st.columns(2)
            
            with col1:
                quantity = st.number_input("Quantité", min_value=1, max_value=10, value=1)
            
            with col2:
                st.write("")  # Spacing
                st.write("")
                if st.button("🛒 " + get_text('add_to_cart'), type="primary", width='stretch'):
                    add_to_cart(product['product_id'], quantity)
                    st.success(f"✅ {quantity} produit(s) ajouté(s) !")
    
    # Section avis clients
    st.markdown("---")
    render_reviews_section(product)
    
    # Recommandations
    st.markdown("---")
    st.markdown(f"## {get_text('recommendations')}")
    
    recommendations = recommendation_engine.get_recommendations(product['product_id'], n_recommendations=4)
    
    # Compléter les champs d'affichage depuis le catalogue (comme dans la grille du catalogue)
    if not recommendations.empty and 'price' not in recommendations.columns:
        catalog = get_products_with_stats()
        extra_fields = ['product_id'] + [c for c in catalog.columns if c not in recommendations.columns]
        recommendations = recommendations.merge(catalog[extra_fields], on='product_id', how='left')
    
    if not recommendations.empty:
        cols = st.columns(4)
        for idx, (_, rec_product) in enumerate(recommendations.iterrows()):
            with cols[idx]:
                render_product_card(rec_product, show_actions=True, key_prefix=f"rec_{product.get('product_id')}_{idx}_")

def render_reviews_section(product):
    """Affiche la section des avis clients avec analyse de sentiment"""
    
    st.markdown(f"## 💬 {get_text('reviews')}")
    
    # Statistiques des avis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Note moyenne", f"{product.get('avg_rating', 0):.1f} / 5")
    
    with col2:
        st.metric("Nombre d'avis", int(product.get('review_count', 0)))
    
    with col3:
        st.metric("Satisfaction", f"{product.get('avg_rating', 0) / 5 * 100:.0f}%")
    
    # Analyse de sentiment (basée sur review_score)
    st.markdown(f"### {get_text('sentiment_analysis')}")

    stats = get_product_review_stats()
    product_stats = stats[stats['product_id'] == product.get('product_id')]

    if not product_stats.empty:
        row = product_stats.iloc[0]
        pos = row.get('positive_pct', 0)
        neu = row.get('neutral_pct', 0)
        neg = row.get('negative_pct', 0)
    else:
        pos, neu, neg = 0, 0, 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="sentiment-positive">
            😊 Positif: {pos:.0f}%
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="sentiment-neutral">
            😐 Neutre: {neu:.0f}%
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="sentiment-negative">
            😞 Négatif: {neg:.0f}%
        </div>
        """, unsafe_allow_html=True)

    # Exemples d'avis réels
    st.markdown("---")

    comments = get_product_review_comments(product.get('product_id'), n=3)

    if not comments:
        st.info("Aucun commentaire client disponible pour ce produit.")
        return

    for review in comments:
        stars = "⭐" * review['rating']

        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #009739;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <strong>Client</strong>
            </div>
            <div style="margin: 0.5rem 0;">
                {stars} {review['rating']}/5
            </div>
            <div style="color: #666; font-size: 0.9rem;">
                {review['text']}
            </div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">
                {review['date']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_product_emoji(category):
    """Retourne un emoji selon la catégorie"""
    
    emoji_map = {
        'health_beauty': '💄',
        'computers_accessories': '💻',
        'auto': '🚗',
        'bed_bath_table': '🛏️',
        'furniture_decor': '🪑',
        'sports_leisure': '⚽',
        'perfumery': '🌸',
        'housewares': '🍳',
        'telephony': '📱',
        'watches_gifts': '⌚',
        'food_drink': '🍔',
        'baby': '👶',
        'stationery': '📚',
        'toys': '🧸',
        'fashion_bags_accessories': '👜',
        'small_appliances': '🔌',
        'electronics': '📺',
        'tools_garden': '🔧',
    }
    
    return emoji_map.get(category, '📦')

def add_to_cart(product_id, quantity=1):
    """Ajoute un produit au panier (stocké en session)"""
    
    if 'cart' not in st.session_state:
        st.session_state.cart = {}
    
    if product_id in st.session_state.cart:
        st.session_state.cart[product_id] += quantity
    else:
        st.session_state.cart[product_id] = quantity

def get_cart_count():
    """Retourne le nombre total de produits dans le panier"""
    
    if 'cart' not in st.session_state:
        return 0
    
    return sum(st.session_state.cart.values())

import pandas as pd
