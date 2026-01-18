"""
🛍️ Olist E-Commerce Analytics Platform
Application Streamlit complète avec interface Admin et Client
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from components.style import load_global_styles
from components.topbar import render_topbar
from components.sidebar import render_sidebar
from utils.data_loader import get_dashboard_kpis, get_sales_over_time, load_reviews

# Configuration de la page
st.set_page_config(
    page_title="Olist Analytics Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS global
load_global_styles()

# Topbar login/logout
render_topbar(key_prefix="app_")

# Sidebar menu
render_sidebar()

# Initialiser la base de données au démarrage
from database.auth_db import init_database, create_default_users
init_database()
create_default_users()

# CSS moderne global (déplacé dans assets/styles.css)

# Vérifier si l'utilisateur est authentifié
if not st.session_state.get('authenticated', False):
    # Page d'accueil pour utilisateurs non connectés
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ Olist Analytics Platform</h1>
        <p>Plateforme d'analyse e-commerce intelligente propulsée par l'IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ **Authentification requise** - Connectez-vous pour accéder à la plateforme")
    
    # Section présentation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: white; 
                    border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🔐</div>
            <h2 style='color: #1e293b; margin-bottom: 1rem;'>Connexion sécurisée</h2>
            <p style='color: #64748b; margin-bottom: 2rem; font-size: 1.1rem;'>
                Accédez à vos données et analyses en temps réel
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Se connecter", type="primary", width='stretch'):
            st.switch_page("pages/0_🔐_Login.py")
        
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem; padding: 1.5rem; 
                    background: #f8fafc; border-radius: 10px;'>
            <p style='color: #64748b; margin-bottom: 1rem;'>📋 <strong>Comptes de démonstration :</strong></p>
            <div style='display: flex; justify-content: space-around; margin-top: 1rem;'>
                <div>
                    <p style='font-weight: 600; color: #667eea;'>👨‍💼 Administrateur</p>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>admin / admin123</p>
                </div>
                <div>
                    <p style='font-weight: 600; color: #f093fb;'>🛒 Client</p>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>client / client123</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Fonctionnalités principales
    st.markdown("### ✨ Fonctionnalités principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Dashboard KPI</div>
            <div class="feature-desc">Visualisez vos indicateurs clés de performance en temps réel avec des graphiques interactifs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🚚</div>
            <div class="feature-title">Prédiction Livraison</div>
            <div class="feature-desc">Estimez les délais de livraison avec précision grâce à nos modèles ML avancés (XGBoost)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Analyse Sentiment</div>
            <div class="feature-desc">Analysez automatiquement les avis clients pour améliorer votre service (NLTK)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistiques de la plateforme
    st.markdown("### 📈 Statistiques de la plateforme")
    
    kpis = get_dashboard_kpis()
    monthly_sales = get_sales_over_time()

    def _pct_delta(current, previous):
        if previous is None or pd.isna(previous) or previous == 0:
            return None
        return (current - previous) / previous * 100

    orders_delta = None
    if len(monthly_sales) >= 2:
        last_orders = monthly_sales.iloc[-1]['order_id']
        prev_orders = monthly_sales.iloc[-2]['order_id']
        orders_delta = _pct_delta(last_orders, prev_orders)

    reviews = load_reviews()
    reviews_monthly = reviews.set_index('review_creation_date').resample('ME')['review_score'].mean().dropna()
    rating_delta = None
    if len(reviews_monthly) >= 2:
        rating_delta = reviews_monthly.iloc[-1] - reviews_monthly.iloc[-2]

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Produits", f"{kpis['total_products']:,}".replace(",", " "))
    with col2:
        st.metric(
            "Commandes",
            f"{kpis['total_orders']:,}".replace(",", " "),
            f"{orders_delta:+.1f}%" if orders_delta is not None else None
        )
    with col3:
        st.metric("Vendeurs", f"{kpis['total_sellers']:,}".replace(",", " "))
    with col4:
        st.metric(
            "Satisfaction",
            f"{kpis['avg_rating']:.2f}/5",
            f"{rating_delta:+.2f} ⭐" if rating_delta is not None else None
        )

else:
    # Utilisateur authentifié - Afficher l'interface selon le rôle
    from components.auth_new import render_user_info
    
    # Afficher les infos utilisateur dans la sidebar
    with st.sidebar:
        render_user_info()
    
    user_name = st.session_state.get('user_full_name', 'Utilisateur')
    user_role = st.session_state.get('user_role', 'client')
    
    # Dashboard selon le rôle
    if user_role == 'admin':
        st.markdown(f"""
        <div class="main-header">
            <h1>👨‍💼 Dashboard Administrateur</h1>
            <p>Bienvenue, {user_name} - Accès complet à toutes les analyses</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ **Accès Administrateur** activé")
        st.info("👈 Utilisez le menu latéral pour naviguer entre les différentes pages d'analyse")
        
        # Accès rapide aux pages
        st.markdown("### 🎯 Accès rapide")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📊</div>
                <h3 style='margin: 0; color: white;'>Dashboard KPI</h3>
                <p style='opacity: 0.9; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Statistiques globales</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Accéder au Dashboard", key="dash", width='stretch'):
                st.switch_page("pages/1_🏠_Dashboard_Admin.py")
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>💬</div>
                <h3 style='margin: 0; color: white;'>Sentiment Analysis</h3>
                <p style='opacity: 0.9; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Analyse des avis</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Accéder à l'Analyse", key="sent", width='stretch'):
                st.switch_page("pages/3_💬_Analyse_Sentiment.py")
        
        with col3:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🚚</div>
                <h3 style='margin: 0; color: white;'>Prédiction Livraison</h3>
                <p style='opacity: 0.9; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Délais de livraison</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Accéder aux Prédictions", key="ship", width='stretch'):
                st.switch_page("pages/2_🚚_Prédiction_Livraison.py")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Pages supplémentaires
        st.markdown("### 🔧 Autres outils")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📦 Prédiction Commandes", width='stretch'):
                st.switch_page("pages/4_📦_Prédiction_Commandes.py")
        
        with col2:
            if st.button("⚙️ Gestion Modèles", width='stretch'):
                st.switch_page("pages/6_⚙️_Gestion_Modèles.py")
        
        with col3:
            if st.button("🛒 Vue Client", width='stretch'):
                st.switch_page("pages/5_🛒_Catalogue_Produits.py")
    
    elif user_role == 'client':
        st.markdown(f"""
        <div class="main-header">
            <h1>🛒 Espace Client</h1>
            <p>Bienvenue, {user_name} - Consultez vos analyses personnalisées</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ **Espace Client** - Accès au catalogue et statistiques personnelles")
        
        # Vue client avec redirection automatique vers le catalogue
        st.info("🔄 Redirection automatique vers le catalogue de produits...")
        st.switch_page("pages/5_🛒_Catalogue_Produits.py")
