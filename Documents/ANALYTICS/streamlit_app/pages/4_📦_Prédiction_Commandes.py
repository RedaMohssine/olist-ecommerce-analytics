"""
📦 Page Prédiction des Commandes Mensuelles
Utilise le modèle XGBoost entraîné dans Nombre_commande_ParMois V111.ipynb
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
from components.charts import create_line_chart, create_bar_chart, create_kpi_chart, create_area_chart
from utils.data_loader import load_orders, load_products, load_order_items
from utils.orders_forecast import get_orders_forecast_model

# Vérification des droits admin
require_admin()

st.set_page_config(page_title="Prédiction Commandes", page_icon="📦", layout="wide")

# CSS global
load_global_styles()

# Topbar login/logout
render_topbar(key_prefix="orders_")

# Sidebar menu
render_sidebar()

# CSS Custom
st.markdown("""
<style>
    .forecast-box {
        background: linear-gradient(135deg, #002776 0%, #003da5 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 39, 118, 0.3);
    }
    .forecast-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .trend-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 1rem;
    }
    .comparison-card {
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
    <h1 style='margin: 0;'>📦 Prédiction des Commandes</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
        Prévision de la demande mensuelle par produit
    </p>
</div>
""", unsafe_allow_html=True)

# ========================================
# CHARGEMENT DU MODÈLE
# ========================================
forecast_model = get_orders_forecast_model()
model_info = forecast_model.get_model_info()

# ========================================
# INDICATEURS DU MODÈLE
# ========================================
st.markdown("## 📊 Performance du Modèle XGBoost")

if forecast_model.is_model_loaded():
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
            "MAE (quantité)",
            f"{mae:.4f}",
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
        train_size = data_info.get('train_size', 0)
        st.markdown(create_kpi_chart(
            "Échantillons",
            f"{train_size:,}",
            "📊"
        ), unsafe_allow_html=True)
    
    # Afficher les features les plus importantes
    with st.expander("🔍 Top Features Importantes"):
        feature_importance = forecast_model.get_feature_importance(top_n=10)
        if feature_importance is not None:
            st.dataframe(
                feature_importance.style.format({'importance': '{:.4f}'}),
                hide_index=True,
                width='stretch'
            )
else:
    st.error("❌ Modèle XGBoost non disponible. Veuillez exécuter le notebook Nombre_commande_ParMois V111.ipynb")
    st.stop()

st.markdown("---")

# ========================================
# MODE DE PRÉDICTION
# ========================================
st.markdown("## 🎯 Mode de Prédiction")

mode = st.radio(
    "Choisissez le mode",
    ["🔮 Prédiction Produit", "📊 Prédiction Globale", "📈 Analyse Tendances", "🎯 Planification Stock"],
    horizontal=True
)

# ========================================
# MODE 1: PRÉDICTION PRODUIT
# ========================================
if mode == "🔮 Prédiction Produit":
    st.markdown("### 🔮 Prédiction par Produit avec XGBoost")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📝 Paramètres de Prédiction")
        
        # Chargement des données pour analyse
        products = load_products()
        order_items = load_order_items()
        orders = load_orders()
        
        if products is not None and order_items is not None and orders is not None:
            # Préparer les données réelles
            orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
            
            # Joindre les données
            order_with_items = order_items.merge(orders[['order_id', 'order_purchase_timestamp']], on='order_id')
            order_with_items = order_with_items.merge(products, on='product_id', how='left')
            
            # Extraire le mois
            order_with_items['month'] = pd.to_datetime(order_with_items['order_purchase_timestamp']).dt.to_period('M')
            
            # Agréger par produit et mois
            product_monthly = order_with_items.groupby(['product_id', 'month']).agg({
                'order_item_id': 'count',  # quantité vendue
                'price': 'mean',
                'freight_value': 'mean',
                'product_weight_g': 'first',
                'product_length_cm': 'first',
                'product_height_cm': 'first',
                'product_width_cm': 'first',
                'product_category_name': 'first'
            }).reset_index()
            
            product_monthly.columns = ['product_id', 'month', 'quantity_sold', 'avg_price', 
                                       'avg_freight', 'weight_g', 'length_cm', 'height_cm', 'width_cm', 'category']
            
            # Calculer le volume
            product_monthly['volume_cm3'] = (
                product_monthly['length_cm'] * 
                product_monthly['height_cm'] * 
                product_monthly['width_cm']
            )
            
            # Sélection du produit
            top_products = product_monthly.groupby('product_id')['quantity_sold'].sum().nlargest(100)
            
            col_input1, col_input2 = st.columns(2)
            
            with col_input1:
                # Sélection du produit
                product_options = []
                for pid in top_products.index[:20]:  # Top 20 produits
                    prod_data = product_monthly[product_monthly['product_id'] == pid].iloc[0]
                    cat = prod_data['category'] if pd.notna(prod_data['category']) else 'Unknown'
                    total_sales = top_products[pid]
                    product_options.append(f"{cat[:20]} (ID: {pid[:8]}...) - {total_sales:.0f} ventes")
                
                selected_product_display = st.selectbox(
                    "Sélectionner un Produit",
                    options=product_options,
                    key='selected_product'
                )
                
                # Extraire l'ID du produit
                selected_product_id = top_products.index[product_options.index(selected_product_display)]
                
                # Mois à prédire
                n_months_future = st.slider(
                    "Nombre de mois à prédire",
                    min_value=1,
                    max_value=12,
                    value=6,
                    key='n_months'
                )
            
            with col_input2:
                # Afficher les infos du produit
                product_data = product_monthly[product_monthly['product_id'] == selected_product_id].iloc[-1]
                
                st.markdown("##### 📊 Caractéristiques")
                st.write(f"**Prix moyen**: R$ {product_data['avg_price']:.2f}")
                st.write(f"**Poids**: {product_data['weight_g']:.0f}g")
                st.write(f"**Volume**: {product_data['volume_cm3']:.0f} cm³")
                st.write(f"**Catégorie**: {product_data['category'] if pd.notna(product_data['category']) else 'N/A'}")
            
            st.markdown("---")
            
            # Historique des ventes
            st.markdown("#### 📊 Historique des Ventes")
            
            product_history = product_monthly[product_monthly['product_id'] == selected_product_id].copy()
            product_history['month_str'] = product_history['month'].astype(str)
            
            chart = create_line_chart(
                product_history.tail(12),
                'month_str',
                'quantity_sold',
                f"Ventes Mensuelles - 12 derniers mois"
            )
            st.plotly_chart(chart, width='stretch')
            
            # Calculer les features historiques
            product_history_sorted = product_history.sort_values('month')
            
            # Lags
            lags = {}
            for lag in [1, 2, 3, 6, 12]:
                if len(product_history_sorted) >= lag:
                    lags[f'lag_{lag}'] = product_history_sorted['quantity_sold'].iloc[-lag]
                else:
                    lags[f'lag_{lag}'] = 0
            
            # Rolling means
            for window in [3, 6, 12]:
                if len(product_history_sorted) >= window:
                    lags[f'rolling_mean_{window}'] = product_history_sorted['quantity_sold'].tail(window).mean()
                    lags[f'rolling_std_{window}'] = product_history_sorted['quantity_sold'].tail(window).std()
                else:
                    lags[f'rolling_mean_{window}'] = 0
                    lags[f'rolling_std_{window}'] = 0
            
            # Autres features
            lags['cumulative_sales_past'] = product_history_sorted['quantity_sold'].sum()
            lags['lifetime_avg_sales'] = product_history_sorted['quantity_sold'].mean()
            lags['cv_3m'] = (lags['rolling_std_3'] / (lags['rolling_mean_3'] + 1e-6)) if lags['rolling_mean_3'] > 0 else 0
            
            # Calcul de la tendance
            if len(product_history_sorted) >= 3:
                recent = product_history_sorted['quantity_sold'].tail(3).values
                lags['trend_3m'] = (recent[-1] - recent[0]) / (len(recent) + 1e-6)
            else:
                lags['trend_3m'] = 0
            
            # Stats par catégorie
            category_stats = product_monthly[
                product_monthly['category'] == product_data['category']
            ]['quantity_sold'].agg(['mean', 'std'])
            
            lags['category_avg_sales'] = category_stats['mean'] if not pd.isna(category_stats['mean']) else 1
            lags['category_std_sales'] = category_stats['std'] if not pd.isna(category_stats['std']) else 0
            lags['sales_vs_category'] = lags['lifetime_avg_sales'] / (lags['category_avg_sales'] + 1e-6)
            
            # Age du produit
            lags['product_age_months'] = len(product_history_sorted)
            lags['cumulative_sales'] = lags['cumulative_sales_past']
            lags['had_sales_last_year'] = 1 if len(product_history_sorted) >= 12 and product_history_sorted['quantity_sold'].iloc[-12] > 0 else 0
            lags['sales_vs_lifetime_avg'] = (
                product_history_sorted['quantity_sold'].iloc[-1] / (lags['lifetime_avg_sales'] + 1e-6)
                if len(product_history_sorted) > 0 else 1
            )
            lags['rolling3_x_trend'] = lags['rolling_mean_3'] * lags['trend_3m']
            
            # Bouton de prédiction
            if st.button("🚀 Prédire avec XGBoost", type="primary", width='stretch'):
                with st.spinner("🔄 Calcul avec XGBoost..."):
                    try:
                        # Préparer les données du produit
                        prod_dict = {
                            'price': product_data['avg_price'],
                            'freight_value': product_data['avg_freight'],
                            'weight_g': product_data['weight_g'],
                            'volume_cm3': product_data['volume_cm3'],
                            'review_score': 4.0,  # Valeur par défaut
                            'payment_value': product_data['avg_price']
                        }
                        
                        # Date de départ
                        last_month = product_history_sorted['month'].max().to_timestamp()
                        start_month = last_month + timedelta(days=32)
                        start_month = start_month.replace(day=1)
                        
                        # Prédire pour plusieurs mois
                        predictions_df = forecast_model.predict_multiple_months(
                            product_data=prod_dict,
                            start_month=start_month,
                            n_months=n_months_future,
                            historical_sales=lags
                        )
                        
                        if predictions_df is not None:
                            st.session_state['predictions_df'] = predictions_df
                            st.session_state['product_history'] = product_history
                            st.success(f"✅ Prédictions générées pour {n_months_future} mois")
                        else:
                            st.error("❌ Erreur lors de la prédiction")
                    
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        else:
            st.error("❌ Données non disponibles")
    
    with col2:
        st.markdown("#### 📊 Résultats")
        
        if 'predictions_df' in st.session_state:
            predictions_df = st.session_state['predictions_df']
            product_history = st.session_state['product_history']
            
            # Moyenne des prédictions
            avg_pred = predictions_df['predicted_quantity'].mean()
            last_actual = product_history['quantity_sold'].iloc[-1]
            
            # Boîte de prédiction
            trend_pct = ((avg_pred / last_actual) - 1) * 100 if last_actual > 0 else 0
            
            st.markdown(f"""
            <div class='forecast-box'>
                <h3 style='margin: 0;'>Demande Moyenne</h3>
                <div class='forecast-value'>{avg_pred:.1f}</div>
                <p style='margin: 0.5rem 0;'>unités/mois</p>
                <div class='trend-badge'>
                    {'📈' if trend_pct > 0 else '📉'} {trend_pct:+.1f}% vs dernier mois
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Détails mensuels
            st.markdown("##### 📅 Prévisions Détaillées")
            
            for idx, row in predictions_df.iterrows():
                month_name = row['month'].strftime('%B %Y')
                pred_qty = row['predicted_quantity']
                
                # Intervalle de confiance (±20%)
                conf_low = pred_qty * 0.8
                conf_high = pred_qty * 1.2
                
                st.markdown(f"""
                <div class='comparison-card'>
                    <strong>{month_name}</strong>
                    <h3 style='margin: 0.5rem 0; color: #002776;'>{pred_qty:.0f} unités</h3>
                    <p style='margin: 0; font-size: 0.9rem; color: #666;'>
                        Intervalle: {conf_low:.0f} - {conf_high:.0f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Actions recommandées
            st.markdown("##### 💡 Recommandations")
            
            if avg_pred > last_actual * 1.2:
                st.success("✅ Augmenter le stock de 20%")
            elif avg_pred < last_actual * 0.8:
                st.warning("⚠️ Réduire le stock de 20%")
            else:
                st.info("ℹ️ Maintenir le niveau actuel")
        
        else:
            st.info("👈 Sélectionnez un produit et cliquez sur 'Prédire'")
            
            st.markdown("##### 💡 Informations")
            st.markdown("""
            - **Modèle**: XGBoost optimisé
            - **Features**: 40 variables
            - **MAE**: 0.0775 unités
            - **R²**: 0.88 (88% de précision)
            
            Le modèle utilise:
            - Historique des ventes (lags)
            - Moyennes mobiles
            - Tendances
            - Saisonnalité
            - Caractéristiques produit
            """)

# ========================================
# MODE 2: PRÉDICTION GLOBALE
# ========================================
elif mode == "📊 Prédiction Globale":
    st.markdown("### 📊 Prédictions Globales Multi-Produits")
    
    orders = load_orders()
    
    if orders is not None:
        # Analyse historique globale
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        monthly_orders = orders.set_index('order_purchase_timestamp').resample('ME').size()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_month = monthly_orders.iloc[-1]
            st.metric("Mois Actuel", f"{current_month:,}")
        
        with col2:
            avg_monthly = monthly_orders.mean()
            st.metric("Moyenne Mensuelle", f"{avg_monthly:.0f}")
        
        with col3:
            trend = ((monthly_orders.iloc[-1] / monthly_orders.iloc[-6]) - 1) * 100
            st.metric("Tendance 6M", f"{trend:+.1f}%")
        
        with col4:
            max_month = monthly_orders.max()
            st.metric("Record", f"{max_month:,}")
        
        st.markdown("---")
        
        # Graphique historique + prédictions
        st.markdown("#### 📈 Historique et Prévisions")
        
        # Génération de prédictions futures (6 mois) via tendance linéaire
        last_value = monthly_orders.iloc[-1]
        future_dates = pd.date_range(start=monthly_orders.index[-1] + timedelta(days=30), periods=6, freq='M')
        
        y = monthly_orders.values
        x = np.arange(len(y))
        if len(y) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            future_x = np.arange(len(y), len(y) + 6)
            predictions = np.maximum(0, slope * future_x + intercept)
        else:
            predictions = [last_value for _ in range(6)]
        
        # Combiner historique et prédictions
        combined_dates = list(monthly_orders.index[-12:]) + list(future_dates)
        combined_values = list(monthly_orders.iloc[-12:]) + predictions
        
        combined_df = pd.DataFrame({
            'Mois': combined_dates,
            'Commandes': combined_values,
            'Type': ['Historique'] * 12 + ['Prédiction'] * 6
        })
        
        chart = create_area_chart(combined_df, 'Mois', 'Commandes', "Commandes Mensuelles - Historique + Prédictions")
        st.plotly_chart(chart, width='stretch')
        
        # Top catégories à suivre (data-driven)
        st.markdown("#### 🎯 Top Catégories à Surveiller")
        
        order_items = load_order_items()
        products = load_products()
        items_with_date = order_items.merge(
            orders[['order_id', 'order_purchase_timestamp']],
            on='order_id',
            how='left'
        ).merge(
            products[['product_id', 'product_category_name_english']],
            on='product_id',
            how='left'
        )
        
        last_date = orders['order_purchase_timestamp'].max()
        last_6m = last_date - pd.DateOffset(months=6)
        prev_6m = last_date - pd.DateOffset(months=12)
        
        recent = items_with_date[items_with_date['order_purchase_timestamp'] >= last_6m]
        prev = items_with_date[
            (items_with_date['order_purchase_timestamp'] >= prev_6m) &
            (items_with_date['order_purchase_timestamp'] < last_6m)
        ]
        
        recent_counts = recent.groupby('product_category_name_english').size()
        prev_counts = prev.groupby('product_category_name_english').size()
        
        growth = ((recent_counts - prev_counts) / prev_counts.replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)
        growth = growth.dropna().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("##### 📈 En Croissance")
            growth_data = growth.head(4).reset_index()
            growth_data.columns = ['Catégorie', 'Croissance (%)']
            st.dataframe(growth_data, hide_index=True, width='stretch')
        
        with col2:
            st.error("##### 📉 En Déclin")
            decline_data = growth.sort_values().head(4).reset_index()
            decline_data.columns = ['Catégorie', 'Croissance (%)']
            st.dataframe(decline_data, hide_index=True, width='stretch')
    
    else:
        st.error("❌ Données non disponibles")

# ========================================
# MODE 3: ANALYSE TENDANCES
# ========================================
elif mode == "📈 Analyse Tendances":
    st.markdown("### 📈 Analyse des Tendances de Vente")
    
    orders = load_orders()
    order_items = load_order_items()
    
    if orders is not None and order_items is not None:
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        
        # Tendance par jour de la semaine
        st.markdown("#### 📅 Ventes par Jour de la Semaine")
        
        orders['day_of_week'] = orders['order_purchase_timestamp'].dt.day_name()
        weekday_sales = orders.groupby('day_of_week').size().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        chart = create_bar_chart(
            pd.DataFrame({
                'Jour': ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'],
                'Commandes': weekday_sales.values
            }),
            'Jour',
            'Commandes',
            "Répartition Hebdomadaire"
        )
        st.plotly_chart(chart, width='stretch')
        
        # Tendance par heure
        st.markdown("#### ⏰ Ventes par Heure de la Journée")
        
        orders['hour'] = orders['order_purchase_timestamp'].dt.hour
        hourly_sales = orders.groupby('hour').size()
        
        chart2 = create_line_chart(
            pd.DataFrame({
                'Heure': hourly_sales.index,
                'Commandes': hourly_sales.values
            }),
            'Heure',
            'Commandes',
            "Pic d'Activité Journalier"
        )
        st.plotly_chart(chart2, width='stretch')
        
        # Saisonnalité mensuelle
        st.markdown("#### 🗓️ Saisonnalité Annuelle")
        
        orders['month'] = orders['order_purchase_timestamp'].dt.month
        monthly_pattern = orders.groupby('month').size()
        
        month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        
        chart3 = create_bar_chart(
            pd.DataFrame({
                'Mois': month_names,
                'Commandes': monthly_pattern.values
            }),
            'Mois',
            'Commandes',
            "Pattern Saisonnier"
        )
        st.plotly_chart(chart3, width='stretch')
        
        # Insights
        st.markdown("#### 💡 Insights Clés")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **🎯 Meilleurs Jours:**
            - Lundi et Mardi
            - Augmentation de 15% vs moyenne
            
            **⏰ Heures de Pointe:**
            - 10h-12h et 14h-16h
            - Optimiser la capacité serveur
            """)
        
        with col2:
            st.success("""
            **📅 Saisonnalité:**
            - Pic en Novembre (Black Friday)
            - Creux en Janvier-Février
            
            **📈 Recommandations:**
            - Stock supplémentaire Q4
            - Promotions en période creuse
            """)
    
    else:
        st.error("❌ Données non disponibles")

# ========================================
# MODE 4: PLANIFICATION STOCK
# ========================================
elif mode == "🎯 Planification Stock":
    st.markdown("### 🎯 Planification et Optimisation du Stock")
    
    orders = load_orders()
    order_items = load_order_items()
    products = load_products()
    
    last_date = orders['order_purchase_timestamp'].max()
    recent_90 = last_date - pd.Timedelta(days=90)
    prev_90 = last_date - pd.Timedelta(days=180)
    
    items_with_date = order_items.merge(
        orders[['order_id', 'order_purchase_timestamp']],
        on='order_id',
        how='left'
    )
    
    recent_sales = items_with_date[items_with_date['order_purchase_timestamp'] >= recent_90]
    prev_sales = items_with_date[
        (items_with_date['order_purchase_timestamp'] >= prev_90) &
        (items_with_date['order_purchase_timestamp'] < recent_90)
    ]
    
    recent_counts = recent_sales.groupby('product_id').size().rename('sales_recent')
    prev_counts = prev_sales.groupby('product_id').size().rename('sales_prev')
    
    velocity = pd.concat([recent_counts, prev_counts], axis=1).fillna(0)
    velocity['growth_pct'] = ((velocity['sales_recent'] - velocity['sales_prev']) / velocity['sales_prev'].replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📦 Produits Actifs (90j)")
        st.metric("Produits", f"{int((velocity['sales_recent'] > 0).sum()):,}")
    
    with col2:
        st.markdown("##### 📈 Ventes 90j")
        st.metric("Total ventes", f"{int(velocity['sales_recent'].sum()):,}")
    
    with col3:
        st.markdown("##### ⚠️ Produits en déclin")
        st.metric("Déclin", f"{int((velocity['growth_pct'] < -20).sum()):,}")
    
    st.markdown("---")
    
    st.markdown("##### 🔔 Produits Nécessitant une Action")
    
    alerts = velocity.copy()
    alerts = alerts.merge(products[['product_id', 'product_category_name_english']], on='product_id', how='left')
    alerts = alerts.sort_values('growth_pct', ascending=False)
    
    to_restock = alerts.head(5)[['product_category_name_english', 'sales_recent', 'growth_pct']].copy()
    to_restock['Action'] = '🔴 Réapprovisionner'
    
    to_reduce = alerts.tail(5)[['product_category_name_english', 'sales_recent', 'growth_pct']].copy()
    to_reduce['Action'] = '⚠️ Réduire'
    
    stock_alerts = pd.concat([to_restock, to_reduce], axis=0)
    stock_alerts.columns = ['Catégorie', 'Ventes (90j)', 'Croissance (%)', 'Action']
    
    st.dataframe(stock_alerts, hide_index=True, width='stretch')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>📦 Prédiction des Commandes</p>
    <p>💡 Astuce: Une bonne prévision = moins de ruptures + moins de surstock</p>
</div>
""", unsafe_allow_html=True)
