"""
📊 Dashboard Admin - Vue d'ensemble KPI et analytics
"""

import streamlit as st
from components.style import load_global_styles
from components.topbar import render_topbar
from components.sidebar import render_sidebar
import pandas as pd
import numpy as np
from components.auth import require_admin
from components.charts import *
from utils.data_loader import (
    get_dashboard_kpis,
    get_sales_over_time,
    get_category_performance,
    get_geographic_distribution,
    load_reviews,
    load_orders,
    load_order_items
)

# Vérification des droits admin
require_admin()

st.set_page_config(page_title="Dashboard Admin", page_icon="📊", layout="wide")

# CSS global
load_global_styles()

# Topbar login/logout
render_topbar(key_prefix="dash_")

# Sidebar menu
render_sidebar()

# Header
st.markdown("""
<div class='admin-header'>
    <h1 style='margin: 0;'>📊 Dashboard Administrateur</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
        Vue d'ensemble des performances de la plateforme
    </p>
</div>
""", unsafe_allow_html=True)

# Chargement des données
with st.spinner("🔄 Chargement des données..."):
    kpis = get_dashboard_kpis()
    sales_over_time = get_sales_over_time()
    category_perf = get_category_performance()
    geo_dist = get_geographic_distribution()
    orders = load_orders()
    order_items = load_order_items()
    reviews = load_reviews()

# Calculs de tendances mensuelles
sales_over_time = sales_over_time.sort_values('month')
if len(sales_over_time) >= 2:
    last = sales_over_time.iloc[-1]
    prev = sales_over_time.iloc[-2]
    orders_growth = ((last['order_id'] - prev['order_id']) / prev['order_id'] * 100) if prev['order_id'] > 0 else 0
    revenue_growth = ((last['payment_value'] - prev['payment_value']) / prev['payment_value'] * 100) if prev['payment_value'] > 0 else 0
else:
    orders_growth = 0
    revenue_growth = 0

# Nouveaux clients par mois (premier achat)
first_purchase = orders.groupby('customer_id')['order_purchase_timestamp'].min()
new_customers_by_month = first_purchase.dt.to_period('M').value_counts().sort_index()
new_customers_growth = 0
if len(new_customers_by_month) >= 2:
    last_new = new_customers_by_month.iloc[-1]
    prev_new = new_customers_by_month.iloc[-2]
    new_customers_growth = ((last_new - prev_new) / prev_new * 100) if prev_new > 0 else 0

# Nouveaux vendeurs par mois (première vente)
items_orders = order_items.merge(
    orders[['order_id', 'order_purchase_timestamp']],
    on='order_id',
    how='left'
)
first_sale = items_orders.groupby('seller_id')['order_purchase_timestamp'].min()
new_sellers_by_month = first_sale.dt.to_period('M').value_counts().sort_index()
new_sellers_growth = 0
if len(new_sellers_by_month) >= 2:
    last_new_sellers = new_sellers_by_month.iloc[-1]
    prev_new_sellers = new_sellers_by_month.iloc[-2]
    new_sellers_growth = ((last_new_sellers - prev_new_sellers) / prev_new_sellers * 100) if prev_new_sellers > 0 else 0

# Tendance de satisfaction (note moyenne mensuelle)
reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
monthly_rating = reviews.set_index('review_creation_date').resample('ME')['review_score'].mean().dropna()
rating_growth = 0
if len(monthly_rating) >= 2:
    rating_growth = monthly_rating.iloc[-1] - monthly_rating.iloc[-2]

# ========================================
# SECTION 1: KPIs PRINCIPAUX
# ========================================
st.markdown("## 📈 Indicateurs Clés")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(create_kpi_chart(
        "Commandes Totales",
        kpis['total_orders'],
        delta=orders_growth,
        suffix=""
    ), unsafe_allow_html=True)

with col2:
    st.markdown(create_kpi_chart(
        "Chiffre d'Affaires",
        kpis['total_revenue'],
        delta=revenue_growth,
        prefix="R$ ",
        suffix=""
    ), unsafe_allow_html=True)

with col3:
    st.markdown(create_kpi_chart(
        "Clients Actifs",
        kpis['total_customers'],
        delta=new_customers_growth,
        suffix=""
    ), unsafe_allow_html=True)

with col4:
    st.markdown(create_kpi_chart(
        "Vendeurs",
        kpis['total_sellers'],
        delta=new_sellers_growth,
        suffix=""
    ), unsafe_allow_html=True)

with col5:
    st.markdown(create_kpi_chart(
        "Note Moyenne",
        f"{kpis['avg_rating']:.2f}",
        prefix="",
        suffix=" / 5"
    ), unsafe_allow_html=True)

st.markdown("---")

# ========================================
# SECTION 2: ÉVOLUTION TEMPORELLE
# ========================================
st.markdown("## 📅 Évolution dans le Temps")

col1, col2 = st.columns(2)

with col1:
    # Ventes dans le temps
    fig_sales = create_area_chart(
        sales_over_time,
        x='month',
        y='order_id',
        title='📦 Nombre de Commandes par Mois'
    )
    st.plotly_chart(fig_sales, width='stretch')

with col2:
    # Revenu dans le temps
    fig_revenue = create_line_chart(
        sales_over_time,
        x='month',
        y='payment_value',
        title='💰 Chiffre d\'Affaires par Mois'
    )
    st.plotly_chart(fig_revenue, width='stretch')

st.markdown("---")

# ========================================
# SECTION 3: PERFORMANCE PAR CATÉGORIE
# ========================================
st.markdown("## 🏷️ Performance par Catégorie")

bottom_categories = category_perf.tail(3)['category'].tolist() if not category_perf.empty else []

col1, col2 = st.columns([2, 1])

with col1:
    # Top catégories
    top_categories = category_perf.head(15)
    fig_cat = create_bar_chart(
        top_categories,
        x='category',
        y='revenue',
        title='💵 Top 15 Catégories par Revenus',
        orientation='h'
    )
    st.plotly_chart(fig_cat, width='stretch')

with col2:
    # Distribution des commandes
    top_5 = category_perf.head(5)
    fig_pie = create_pie_chart(
        top_5,
        values='orders',
        names='category',
        title='📊 Part des Commandes (Top 5)'
    )
    st.plotly_chart(fig_pie, width='stretch')

# Tableau détaillé
with st.expander("📋 Tableau Détaillé des Catégories"):
    category_display = category_perf.copy()
    category_display['revenue'] = category_display['revenue'].apply(lambda x: f"R$ {x:,.2f}")
    category_display.columns = ['Catégorie', 'Commandes', 'Revenus']
    st.dataframe(category_display, width='stretch', height=400)

st.markdown("---")

# ========================================
# SECTION 4: DISTRIBUTION GÉOGRAPHIQUE
# ========================================
st.markdown("## 🗺️ Distribution Géographique")

col1, col2 = st.columns([2, 1])

with col1:
    # Top états
    top_states = geo_dist.head(15)
    fig_geo = create_bar_chart(
        top_states,
        x='state',
        y='orders',
        title='📍 Top 15 États par Nombre de Commandes',
        orientation='v'
    )
    st.plotly_chart(fig_geo, width='stretch')

with col2:
    # Métriques géographiques
    st.markdown("### 📊 Statistiques Géo")
    
    st.metric("États Couverts", len(geo_dist))
    st.metric("État Principal", geo_dist.iloc[0]['state'])
    st.metric("Commandes État #1", f"{geo_dist.iloc[0]['orders']:,}")
    
    # Concentration
    top_5_concentration = (geo_dist.head(5)['orders'].sum() / geo_dist['orders'].sum() * 100)
    st.metric("Concentration Top 5", f"{top_5_concentration:.1f}%")

st.markdown("---")

# ========================================
# SECTION 5: ANALYSE DES AVIS
# ========================================
st.markdown("## 💬 Analyse des Avis Clients")

reviews = load_reviews()

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_reviews = len(reviews)
    st.metric("Total Avis", f"{total_reviews:,}")

with col2:
    avg_score = reviews['review_score'].mean()
    st.metric("Note Moyenne", f"{avg_score:.2f} / 5")

with col3:
    positive_reviews = len(reviews[reviews['review_score'] >= 4])
    positive_pct = (positive_reviews / total_reviews * 100)
    st.metric("Avis Positifs", f"{positive_pct:.1f}%")

with col4:
    negative_reviews = len(reviews[reviews['review_score'] <= 2])
    negative_pct = (negative_reviews / total_reviews * 100)
    st.metric("Avis Négatifs", f"{negative_pct:.1f}%")

# Distribution des notes
reviews_dist = reviews['review_score'].value_counts().sort_index()
reviews_df = pd.DataFrame({
    'note': reviews_dist.index,
    'count': reviews_dist.values
})

fig_reviews = create_bar_chart(
    reviews_df,
    x='note',
    y='count',
    title='⭐ Distribution des Notes Client',
    orientation='v'
)
st.plotly_chart(fig_reviews, width='stretch')

st.markdown("---")

# ========================================
# SECTION 6: PERFORMANCE DE LIVRAISON
# ========================================
st.markdown("## 🚚 Performance de Livraison")

orders = load_orders()

# Calculer les délais
orders['delivery_time'] = (
    orders['order_delivered_customer_date'] - 
    orders['order_purchase_timestamp']
).dt.days

delivered_orders = orders[orders['order_status'] == 'delivered'].copy()

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_delivery = delivered_orders['delivery_time'].mean()
    st.metric("Délai Moyen", f"{avg_delivery:.1f} jours")

with col2:
    median_delivery = delivered_orders['delivery_time'].median()
    st.metric("Délai Médian", f"{median_delivery:.0f} jours")

with col3:
    on_time = len(orders[
        (orders['order_delivered_customer_date'] <= orders['order_estimated_delivery_date']) &
        (orders['order_status'] == 'delivered')
    ])
    on_time_pct = (on_time / len(delivered_orders) * 100)
    st.metric("Livraisons à Temps", f"{on_time_pct:.1f}%")

with col4:
    delivery_rate = (len(delivered_orders) / len(orders) * 100)
    st.metric("Taux de Livraison", f"{delivery_rate:.1f}%")

# Distribution des délais
delivery_dist = delivered_orders['delivery_time'].value_counts().sort_index()
delivery_df = pd.DataFrame({
    'days': delivery_dist.index,
    'count': delivery_dist.values
})
delivery_df = delivery_df[delivery_df['days'] <= 60]  # Filtre outliers

fig_delivery = create_area_chart(
    delivery_df,
    x='days',
    y='count',
    title='📦 Distribution des Délais de Livraison'
)
st.plotly_chart(fig_delivery, width='stretch')

st.markdown("---")

# ========================================
# SECTION 7: TENDANCES ET INSIGHTS
# ========================================
st.markdown("## 🔍 Insights et Tendances")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Tendances Positives")
    st.success(f"✅ Croissance des ventes de {orders_growth:+.1f}% ce mois")
    st.success(f"✅ Variation note moyenne: {rating_growth:+.2f} points")
    st.success(f"✅ Nouveaux vendeurs: {new_sellers_by_month.iloc[-1] if len(new_sellers_by_month) else 0}")
    st.success(f"✅ Avis positifs: {positive_pct:.1f}%")

with col2:
    st.markdown("### ⚠️ Points d'Attention")
    st.warning(f"⚠️ Avis négatifs: {negative_pct:.1f}%")
    st.warning(f"⚠️ Délai moyen: {avg_delivery:.1f} jours")
    st.warning(f"⚠️ Concentration Top 5 états: {top_5_concentration:.1f}%")
    if bottom_categories:
        st.warning(f"⚠️ Catégories faibles: {', '.join(bottom_categories)}")
    else:
        st.warning("⚠️ Catégories sous-performantes à analyser")

st.markdown("---")

# ========================================
# SECTION 8: ACTIONS RAPIDES
# ========================================
st.markdown("## 🚀 Actions Rapides")

col1, col2 = st.columns(2)

with col1:
    report_df = pd.DataFrame([
        {"metric": "total_orders", "value": kpis['total_orders']},
        {"metric": "total_revenue", "value": kpis['total_revenue']},
        {"metric": "total_customers", "value": kpis['total_customers']},
        {"metric": "total_sellers", "value": kpis['total_sellers']},
        {"metric": "avg_rating", "value": kpis['avg_rating']},
        {"metric": "avg_delivery_time", "value": avg_delivery}
    ])
    st.download_button(
        "📊 Exporter Rapport (CSV)",
        report_df.to_csv(index=False),
        "dashboard_report.csv",
        "text/csv",
        width='stretch'
    )

with col2:
    if st.button("🔔 Générer Alertes", width='stretch'):
        alerts = []
        if negative_pct > 15:
            alerts.append("Avis négatifs élevés (>15%)")
        if avg_delivery > 15:
            alerts.append("Délai moyen supérieur à 15 jours")
        if top_5_concentration > 60:
            alerts.append("Forte concentration géographique")
        if not alerts:
            st.success("✅ Aucun signal critique détecté")
        else:
            for a in alerts:
                st.warning(f"⚠️ {a}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>📊 Dashboard mis à jour en temps réel</p>
    <p>Dernière mise à jour: Aujourd'hui</p>
</div>
""", unsafe_allow_html=True)
