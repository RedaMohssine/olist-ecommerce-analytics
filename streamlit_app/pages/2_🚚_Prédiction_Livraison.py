"""
🚚 Page Prédiction du Délai de Livraison
Utilise le modèle XGBoost entraîné dans analytics_shipping_timev2.ipynb
"""

import streamlit as st
from components.style import load_global_styles
from components.topbar import render_topbar
from components.sidebar import render_sidebar
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from components.auth import require_admin
from components.translations import get_text
from components.charts import create_line_chart, create_bar_chart, create_kpi_chart
from utils.data_loader import load_orders, load_customers, load_sellers, load_order_items, load_products
from utils.shipping_forecast import get_shipping_forecast_model

# Vérification des droits admin
require_admin()

st.set_page_config(page_title="Prédiction Livraison", page_icon="🚚", layout="wide")

# CSS global
load_global_styles()

# Topbar login/logout
render_topbar(key_prefix="ship_")

# Sidebar menu
render_sidebar()

# CSS Custom
st.markdown("""
<style>
    .prediction-box {
        background: linear-gradient(135deg, #009739 0%, #00a840 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 151, 57, 0.3);
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 1rem;
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='admin-header'>
    <h1 style='margin: 0;'>🚚 Prédiction du Délai de Livraison</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
        Estimez le temps de livraison avec Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ========================================
# CHARGEMENT DU MODÈLE
# ========================================
shipping_model = get_shipping_forecast_model()
model_info = shipping_model.get_model_info()

# ========================================
# INDICATEURS DU MODÈLE
# ========================================
st.markdown("## 📊 Performance du Modèle XGBoost")

if shipping_model.is_model_loaded():
    metrics = model_info.get('metrics', {})
    data_info = model_info.get('data_info', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        r2_score = metrics.get('r2_score', 0)
        st.markdown(create_kpi_chart(
            "R² Score",
            f"{r2_score:.4f}",
            "🎯"
        ), unsafe_allow_html=True)
    
    with col2:
        mae = metrics.get('mae', 0)
        st.markdown(create_kpi_chart(
            "MAE (jours)",
            f"{mae:.2f}",
            "📏"
        ), unsafe_allow_html=True)
    
    with col3:
        n_features = data_info.get('n_features', 0)
        st.markdown(create_kpi_chart(
            "Features",
            str(n_features),
            "🔢"
        ), unsafe_allow_html=True)
    
    with col4:
        rmse = metrics.get('rmse', 0)
        st.markdown(create_kpi_chart(
            "RMSE",
            f"{rmse:.2f}j",
            "📊"
        ), unsafe_allow_html=True)
else:
    st.error("❌ Modèle XGBoost non disponible. Veuillez exécuter le notebook analytics_shipping_timev2.ipynb")
    st.stop()

st.markdown("---")

# ========================================
# MODE DE PRÉDICTION
# ========================================
st.markdown("## 🎯 Mode de Prédiction")

mode = st.radio(
    "Choisissez le mode",
    ["🔮 Prédiction Unique", "📊 Prédiction par Lot", "📈 Analyse Historique"],
    horizontal=True
)

# ========================================
# MODE 1: PRÉDICTION UNIQUE
# ========================================
if mode == "🔮 Prédiction Unique":
    st.markdown("### 🔮 Prédiction Unique")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='input-section'>", unsafe_allow_html=True)
        st.markdown("#### 📝 Informations de la Commande")
        
        # Chargement des données
        customers = load_customers()
        sellers = load_sellers()
        products = load_products()
        
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            # État du client
            customer_states = customers['customer_state'].unique().tolist() if customers is not None else []
            customer_state = st.selectbox(
                "État du Client",
                options=customer_states if customer_states else ['SP', 'RJ', 'MG', 'PR', 'RS'],
                key='customer_state'
            )
            
            # ZIP du client (5 chiffres)
            customer_zip = st.number_input(
                "Code Postal Client (5 chiffres)",
                min_value=1000,
                max_value=99999,
                value=1310,
                step=1,
                key='customer_zip'
            )
        
        with col_input2:
            # Vendeur (simulé)
            if sellers is not None and len(sellers) > 0:
                seller_sample = sellers.sort_values('seller_id').head(min(20, len(sellers)))
                seller_options = [f"{s[:8]}... ({st})" for s, st in zip(seller_sample['seller_id'], seller_sample['seller_state'])]
                selected_seller = st.selectbox(
                    "Sélectionner un Vendeur",
                    options=seller_options,
                    key='seller_option'
                )
                seller_idx = seller_options.index(selected_seller)
                seller_id = seller_sample.iloc[seller_idx]['seller_id']
                seller_state = seller_sample.iloc[seller_idx]['seller_state']
                seller_zip = seller_sample.iloc[seller_idx]['seller_zip_code_prefix']
            else:
                seller_id = 'unknown'
                seller_state = st.selectbox(
                    "État du Vendeur",
                    options=['SP', 'RJ', 'MG', 'PR', 'RS'],
                    key='seller_state'
                )
                seller_zip = st.number_input(
                    "Code Postal Vendeur",
                    min_value=1000,
                    max_value=99999,
                    value=20031,
                    step=1,
                    key='seller_zip'
                )
        
        st.markdown("---")
        
        col_input3, col_input4 = st.columns(2)
        
        with col_input3:
            # Poids du produit
            product_weight = st.number_input(
                "Poids du Produit (g)",
                min_value=0,
                max_value=50000,
                value=500,
                step=100,
                key='product_weight'
            )
            
            # Prix
            price = st.number_input(
                "Prix (R$)",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                step=10.0,
                key='price'
            )
        
        with col_input4:
            # Dimensions
            product_length = st.number_input(
                "Longueur (cm)",
                min_value=0,
                max_value=200,
                value=20,
                step=1,
                key='product_length'
            )
            
            product_height = st.number_input(
                "Hauteur (cm)",
                min_value=0,
                max_value=200,
                value=10,
                step=1,
                key='product_height'
            )
            
            product_width = st.number_input(
                "Largeur (cm)",
                min_value=0,
                max_value=200,
                value=15,
                step=1,
                key='product_width'
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Bouton de prédiction
        if st.button("🚀 Lancer la Prédiction XGBoost", type="primary", width='stretch'):
            with st.spinner("🔄 Calcul avec XGBoost..."):
                try:
                    # Préparer les données pour la prédiction
                    order_data = {
                        'seller_id': seller_id,
                        'customer_state': customer_state,
                        'seller_state': seller_state,
                        'customer_zip_code_prefix': customer_zip,
                        'seller_zip_code_prefix': seller_zip,
                        'product_weight_g': product_weight,
                        'price': price,
                        'product_length_cm': product_length,
                        'product_width_cm': product_width,
                        'product_height_cm': product_height,
                        'freight_value': price * 0.15,  # 15% du prix
                        'num_items': 1,
                        'num_unique_sellers': 1,
                        'total_freight_value': price * 0.15,
                        'total_payment_value': price,
                        'num_payments': 1,
                        'purchase_date': datetime.now(),
                        'product_category_name': 'eletronicos',
                        'customer_state_geo': customer_state,
                        'seller_state_geo': seller_state,
                        'product_name_lenght': 50,
                        'product_description_lenght': 500
                    }
                    
                    # Données de géolocalisation (simplifiées - utilise des coordonnées moyennes par état)
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
                    
                    geolocation_data = {
                        'customer_lat': customer_lat,
                        'customer_lng': customer_lng,
                        'seller_lat': seller_lat,
                        'seller_lng': seller_lng
                    }
                    
                    # Prédiction avec XGBoost
                    prediction_days = shipping_model.predict(order_data, geolocation_data)
                    
                    if prediction_days is not None:
                        # Calculer la distance
                        distance_km = shipping_model.haversine_distance(
                            customer_lat, customer_lng, seller_lat, seller_lng
                        )
                        
                        st.session_state['prediction'] = prediction_days
                        st.session_state['distance'] = distance_km
                        st.session_state['order_data'] = order_data
                        st.success(f"✅ Prédiction réussie: {prediction_days:.1f} jours")
                    else:
                        st.error("❌ Erreur lors de la prédiction")
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        st.markdown("#### 📊 Résultat de la Prédiction")
        
        if 'prediction' in st.session_state:
            prediction_days = st.session_state['prediction']
            distance = st.session_state['distance']
            
            # Boîte de prédiction
            st.markdown(f"""
            <div class='prediction-box'>
                <h3 style='margin: 0;'>Délai Estimé (XGBoost)</h3>
                <div class='prediction-value'>{prediction_days:.1f} jours</div>
                <p style='margin: 0.5rem 0;'>Distance: {distance:.0f} km</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Date de livraison estimée
            order_date = datetime.now()
            estimated_delivery = order_date + timedelta(days=int(prediction_days))
            
            st.success(f"""
            📅 **Date de commande:** {order_date.strftime('%d/%m/%Y')}
            
            🎯 **Livraison estimée:** {estimated_delivery.strftime('%d/%m/%Y')}
            
            📦 **Statut:** {'⚡ Express' if prediction_days < 7 else '🚚 Standard' if prediction_days < 15 else '🐌 Lent'}
            """)
            
            # Facteurs influençant
            st.markdown("##### 🎯 Facteurs Clés")
            
            factors = pd.DataFrame({
                'Facteur': ['Distance', 'Poids', 'Volume', 'État'],
                'Impact': [
                    'Élevé' if distance > 1000 else 'Moyen' if distance > 500 else 'Faible',
                    'Élevé' if product_weight > 5000 else 'Moyen' if product_weight > 1000 else 'Faible',
                    'Moyen',
                    'Faible'
                ]
            })
            
            st.dataframe(factors, hide_index=True, width='stretch')
        
        else:
            st.info("👈 Remplissez le formulaire et cliquez sur 'Lancer la Prédiction'")
            
            st.markdown("##### 💡 Conseils")
            st.markdown("""
            - Vérifiez les codes postaux
            - Le poids influence fortement le délai
            - Distance calculée automatiquement
            - Confiance basée sur l'historique
            """)

# ========================================
# MODE 2: PRÉDICTION PAR LOT
# ========================================
elif mode == "📊 Prédiction par Lot":
    st.markdown("### 📊 Prédiction par Lot (CSV)")
    
    st.info("""
    **Format CSV requis:**
    - customer_state, seller_state
    - product_weight_g, price
    - product_length_cm, product_width_cm, product_height_cm
    """)
    
    uploaded_file = st.file_uploader("📁 Upload fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            st.success(f"✅ {len(df_batch)} lignes chargées")
            
            st.dataframe(df_batch.head(), width='stretch')
            
            if st.button("🚀 Lancer les Prédictions", type="primary"):
                with st.spinner("🔄 Traitement en cours..."):
                    predictions = []
                    distances = []
                    
                    state_coords = {
                        'SP': (-23.55, -46.63),
                        'RJ': (-22.91, -43.17),
                        'MG': (-19.92, -43.94),
                        'PR': (-25.42, -49.27),
                        'RS': (-30.03, -51.23),
                        'BA': (-12.97, -38.51),
                        'SC': (-27.59, -48.55)
                    }
                    
                    for _, row in df_batch.iterrows():
                        customer_state = row.get('customer_state', 'SP')
                        seller_state = row.get('seller_state', 'SP')
                        
                        customer_lat, customer_lng = state_coords.get(customer_state, (-23.55, -46.63))
                        seller_lat, seller_lng = state_coords.get(seller_state, (-23.55, -46.63))
                        
                        order_data = {
                            'seller_id': row.get('seller_id'),
                            'customer_state': customer_state,
                            'seller_state': seller_state,
                            'customer_zip_code_prefix': row.get('customer_zip_code_prefix', 10000),
                            'seller_zip_code_prefix': row.get('seller_zip_code_prefix', 10000),
                            'product_weight_g': row.get('product_weight_g', 500),
                            'price': row.get('price', 100.0),
                            'product_length_cm': row.get('product_length_cm', 20),
                            'product_width_cm': row.get('product_width_cm', 15),
                            'product_height_cm': row.get('product_height_cm', 10),
                            'freight_value': row.get('freight_value', row.get('price', 100.0) * 0.15),
                            'num_items': row.get('num_items', 1),
                            'num_unique_sellers': row.get('num_unique_sellers', 1),
                            'total_freight_value': row.get('total_freight_value', row.get('freight_value', 15.0)),
                            'total_payment_value': row.get('total_payment_value', row.get('price', 100.0)),
                            'num_payments': row.get('num_payments', 1),
                            'purchase_date': pd.Timestamp.now(),
                            'product_category_name': row.get('product_category_name', 'None'),
                            'customer_state_geo': row.get('customer_state_geo', customer_state),
                            'seller_state_geo': row.get('seller_state_geo', seller_state),
                            'product_name_lenght': row.get('product_name_lenght', 50),
                            'product_description_lenght': row.get('product_description_lenght', 500)
                        }
                        
                        geolocation_data = {
                            'customer_lat': customer_lat,
                            'customer_lng': customer_lng,
                            'seller_lat': seller_lat,
                            'seller_lng': seller_lng
                        }
                        
                        pred = shipping_model.predict(order_data, geolocation_data)
                        predictions.append(pred if pred is not None else np.nan)
                        dist = shipping_model.haversine_distance(customer_lat, customer_lng, seller_lat, seller_lng)
                        distances.append(dist)
                    
                    df_batch['predicted_delivery_days'] = predictions
                    df_batch['distance_km'] = distances
                    
                    st.success("✅ Prédictions terminées!")
                    
                    # Statistiques
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Délai Moyen", f"{df_batch['predicted_delivery_days'].mean():.1f} jours")
                    
                    with col2:
                        st.metric("Min", f"{df_batch['predicted_delivery_days'].min():.1f} jours")
                    
                    with col3:
                        st.metric("Max", f"{df_batch['predicted_delivery_days'].max():.1f} jours")
                    
                    # Résultats
                    st.markdown("#### 📋 Résultats")
                    st.dataframe(df_batch, width='stretch')
                    
                    # Export
                    csv = df_batch.to_csv(index=False)
                    st.download_button(
                        "💾 Télécharger les Résultats",
                        csv,
                        "predictions_delivery.csv",
                        "text/csv",
                        width='stretch'
                    )
        
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

# ========================================
# MODE 3: ANALYSE HISTORIQUE
# ========================================
elif mode == "📈 Analyse Historique":
    st.markdown("### 📈 Analyse des Performances Historiques")
    
    orders = load_orders()
    
    if orders is not None:
        # Calcul des délais réels
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        
        delivered_orders = orders[orders['order_status'] == 'delivered'].copy()
        delivered_orders['delivery_days'] = (
            delivered_orders['order_delivered_customer_date'] - 
            delivered_orders['order_purchase_timestamp']
        ).dt.days
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_delivery = delivered_orders['delivery_days'].mean()
            st.metric("Délai Moyen Réel", f"{avg_delivery:.1f} jours")
        
        with col2:
            median_delivery = delivered_orders['delivery_days'].median()
            st.metric("Médiane", f"{median_delivery:.1f} jours")
        
        with col3:
            on_time_rate = (delivered_orders['delivery_days'] <= 10).mean()
            st.metric("Taux < 10 jours", f"{on_time_rate:.1%}")
        
        with col4:
            std_delivery = delivered_orders['delivery_days'].std()
            st.metric("Écart-type", f"{std_delivery:.1f} jours")
        
        st.markdown("---")
        
        # Distribution des délais
        st.markdown("#### 📊 Distribution des Délais de Livraison")
        
        delivery_dist = delivered_orders['delivery_days'].value_counts().sort_index().head(30)
        
        chart = create_bar_chart(
            pd.DataFrame({
                'Jours': delivery_dist.index,
                'Commandes': delivery_dist.values
            }),
            'Jours',
            'Commandes',
            "Distribution des Délais"
        )
        
        st.plotly_chart(chart, width='stretch')
        
        # Évolution temporelle
        st.markdown("#### 📈 Évolution Temporelle")
        
        monthly_delivery = delivered_orders.set_index('order_purchase_timestamp').resample('ME')['delivery_days'].mean()
        
        chart2 = create_line_chart(
            pd.DataFrame({
                'Mois': monthly_delivery.index,
                'Délai Moyen': monthly_delivery.values
            }),
            'Mois',
            'Délai Moyen',
            "Délai Moyen par Mois"
        )
        
        st.plotly_chart(chart2, width='stretch')
    
    else:
        st.error("❌ Données non disponibles")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🚚 Prédiction de Livraison</p>
    <p>💡 Astuce: Plus les informations sont précises, meilleure est la prédiction</p>
</div>
""", unsafe_allow_html=True)
