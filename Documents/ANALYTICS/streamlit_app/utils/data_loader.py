"""
Chargement et gestion des données
"""

import pandas as pd
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "Data"

@st.cache_data(ttl=3600)
def load_products():
    """Charge le dataset des produits"""
    products = pd.read_csv(DATA_PATH / "olist_products_dataset.csv")
    translation = pd.read_csv(DATA_PATH / "product_category_name_translation.csv")
    
    # Fusion avec traduction
    products = products.merge(translation, on='product_category_name', how='left')
    
    return products

@st.cache_data(ttl=3600)
def load_orders():
    """Charge le dataset des commandes"""
    orders = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    
    return orders

@st.cache_data(ttl=3600)
def load_order_items():
    """Charge le dataset des items de commande"""
    return pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")

@st.cache_data(ttl=3600)
def load_customers():
    """Charge le dataset des clients"""
    return pd.read_csv(DATA_PATH / "olist_customers_dataset.csv")

@st.cache_data(ttl=3600)
def load_sellers():
    """Charge le dataset des vendeurs"""
    return pd.read_csv(DATA_PATH / "olist_sellers_dataset.csv")

@st.cache_data(ttl=3600)
def load_reviews():
    """Charge le dataset des avis"""
    reviews = pd.read_csv(DATA_PATH / "olist_order_reviews_dataset.csv")
    reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
    
    return reviews

@st.cache_data(ttl=3600)
def load_payments():
    """Charge le dataset des paiements"""
    return pd.read_csv(DATA_PATH / "olist_order_payments_dataset.csv")

@st.cache_data(ttl=3600)
def get_products_with_stats():
    """Charge les produits avec statistiques agrégées"""
    products = load_products()
    order_items = load_order_items()
    reviews = load_reviews()
    orders = load_orders()
    
    # Fusion items + orders pour avoir les dates
    items_orders = order_items.merge(
        orders[['order_id', 'order_status', 'order_purchase_timestamp']],
        on='order_id',
        how='left'
    )
    
    # Agrégation par produit
    product_stats = items_orders.groupby('product_id').agg({
        'order_id': 'count',
        'price': 'mean',
        'freight_value': 'mean'
    }).rename(columns={'order_id': 'total_sales'}).reset_index()
    
    # Ajouter les notes moyennes
    items_reviews = order_items.merge(
        reviews[['order_id', 'review_score']], 
        on='order_id', 
        how='left'
    )
    
    review_stats = items_reviews.groupby('product_id').agg({
        'review_score': ['mean', 'count']
    }).reset_index()
    review_stats.columns = ['product_id', 'avg_rating', 'review_count']
    
    # Fusion finale
    products_full = products.merge(product_stats, on='product_id', how='left')
    products_full = products_full.merge(review_stats, on='product_id', how='left')
    
    # Remplissage des valeurs manquantes
    products_full['total_sales'] = products_full['total_sales'].fillna(0)
    products_full['avg_rating'] = products_full['avg_rating'].fillna(3.0)
    products_full['review_count'] = products_full['review_count'].fillna(0)
    
    # Statut de stock basé sur l'activité récente (data-driven)
    last_sale = (
        items_orders.groupby('product_id')['order_purchase_timestamp']
        .max()
        .rename('last_order_date')
        .reset_index()
    )
    products_full = products_full.merge(last_sale, on='product_id', how='left')
    max_date = orders['order_purchase_timestamp'].max()
    recent_threshold = max_date - pd.Timedelta(days=90)
    products_full['in_stock'] = products_full['last_order_date'] >= recent_threshold
    products_full['in_stock'] = products_full['in_stock'].fillna(False)
    
    return products_full

@st.cache_data(ttl=3600)
def get_dashboard_kpis():
    """Calcule les KPIs pour le dashboard admin"""
    orders = load_orders()
    order_items = load_order_items()
    customers = load_customers()
    sellers = load_sellers()
    reviews = load_reviews()
    payments = load_payments()
    
    kpis = {
        'total_orders': len(orders),
        'total_revenue': payments['payment_value'].sum(),
        'total_products': len(order_items['product_id'].unique()),
        'total_customers': len(customers),
        'total_sellers': len(sellers),
        'avg_rating': reviews['review_score'].mean(),
        'delivered_orders': len(orders[orders['order_status'] == 'delivered']),
        'avg_delivery_time': (
            orders['order_delivered_customer_date'] - 
            orders['order_purchase_timestamp']
        ).dt.days.mean()
    }
    
    return kpis

@st.cache_data(ttl=3600)
def get_sales_over_time():
    """Calcule les ventes dans le temps"""
    orders = load_orders()
    payments = load_payments()
    
    orders_payments = orders.merge(payments, on='order_id', how='left')
    orders_payments['month'] = orders_payments['order_purchase_timestamp'].dt.to_period('M')
    
    monthly_sales = orders_payments.groupby('month').agg({
        'order_id': 'count',
        'payment_value': 'sum'
    }).reset_index()
    
    monthly_sales['month'] = monthly_sales['month'].dt.to_timestamp()
    
    return monthly_sales

@st.cache_data(ttl=3600)
def get_category_performance():
    """Performance par catégorie"""
    products = load_products()
    order_items = load_order_items()
    translation = pd.read_csv(DATA_PATH / "product_category_name_translation.csv")
    
    items_products = order_items.merge(products[['product_id', 'product_category_name']], on='product_id')
    items_products = items_products.merge(translation, on='product_category_name', how='left')
    
    category_stats = items_products.groupby('product_category_name_english').agg({
        'order_id': 'count',
        'price': 'sum'
    }).reset_index()
    
    category_stats.columns = ['category', 'orders', 'revenue']
    category_stats = category_stats.sort_values('revenue', ascending=False)
    
    return category_stats

@st.cache_data(ttl=3600)
def get_geographic_distribution():
    """Distribution géographique des commandes"""
    orders = load_orders()
    customers = load_customers()
    
    orders_customers = orders.merge(customers, on='customer_id')
    
    state_stats = orders_customers.groupby('customer_state').agg({
        'order_id': 'count'
    }).reset_index()
    
    state_stats.columns = ['state', 'orders']
    state_stats = state_stats.sort_values('orders', ascending=False)
    
    return state_stats

@st.cache_data(ttl=3600)
def get_product_review_stats():
    """Statistiques de sentiment basées sur review_score par produit"""
    order_items = load_order_items()
    reviews = load_reviews()

    items_reviews = order_items.merge(
        reviews[['order_id', 'review_score']],
        on='order_id',
        how='left'
    )

    def classify(score):
        if pd.isna(score):
            return None
        if score >= 4:
            return 'positive'
        if score == 3:
            return 'neutral'
        return 'negative'

    items_reviews['sentiment'] = items_reviews['review_score'].apply(classify)

    counts = (
        items_reviews.groupby(['product_id', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    ratings = items_reviews.groupby('product_id').agg(
        avg_rating=('review_score', 'mean'),
        review_count=('review_score', 'count')
    ).reset_index()

    stats = counts.merge(ratings, on='product_id', how='left')
    for col in ['positive', 'neutral', 'negative']:
        if col not in stats.columns:
            stats[col] = 0

    total = stats[['positive', 'neutral', 'negative']].sum(axis=1).replace(0, 1)
    stats['positive_pct'] = stats['positive'] / total * 100
    stats['neutral_pct'] = stats['neutral'] / total * 100
    stats['negative_pct'] = stats['negative'] / total * 100

    return stats

@st.cache_data(ttl=3600)
def get_product_review_comments(product_id, n=3):
    """Retourne quelques commentaires réels d'avis pour un produit"""
    order_items = load_order_items()
    reviews = load_reviews()

    items_reviews = order_items.merge(
        reviews[['order_id', 'review_score', 'review_creation_date', 'review_comment_message']],
        on='order_id',
        how='left'
    )

    product_reviews = items_reviews[
        (items_reviews['product_id'] == product_id) &
        (items_reviews['review_comment_message'].notna()) &
        (items_reviews['review_comment_message'].str.strip() != '')
    ]

    if product_reviews.empty:
        return []

    product_reviews = product_reviews.sort_values('review_creation_date', ascending=False)
    comments = product_reviews.head(n)

    return [
        {
            'rating': int(row['review_score']) if not pd.isna(row['review_score']) else 0,
            'date': row['review_creation_date'].strftime('%d %b %Y') if not pd.isna(row['review_creation_date']) else '',
            'text': row['review_comment_message']
        }
        for _, row in comments.iterrows()
    ]

@st.cache_data(ttl=3600)
def get_state_options():
    """Liste des états disponibles (clients + vendeurs)"""
    customers = load_customers()
    sellers = load_sellers()
    states = set()
    if customers is not None and 'customer_state' in customers.columns:
        states.update(customers['customer_state'].dropna().unique().tolist())
    if sellers is not None and 'seller_state' in sellers.columns:
        states.update(sellers['seller_state'].dropna().unique().tolist())
    return sorted(states)

@st.cache_data(ttl=3600)
def get_product_seller_state_map():
    """Map product_id -> seller_state (vendeur principal par produit)"""
    order_items = load_order_items()
    sellers = load_sellers()

    if order_items is None or sellers is None:
        return {}

    items_sellers = order_items.merge(
        sellers[['seller_id', 'seller_state']],
        on='seller_id',
        how='left'
    )

    counts = (
        items_sellers.groupby(['product_id', 'seller_state'])
        .size()
        .reset_index(name='count')
    )

    if counts.empty:
        return {}

    idx = counts.groupby('product_id')['count'].idxmax()
    top = counts.loc[idx]

    return dict(zip(top['product_id'], top['seller_state']))
