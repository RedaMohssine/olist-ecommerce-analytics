"""
Composant de navigation dans la sidebar
"""

import streamlit as st
from components.translations import get_text

def render_sidebar():
    """Affiche le menu de navigation selon le rôle utilisateur"""
    
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        user_role = st.session_state.get('user_role')
        
        if user_role == 'admin':
            render_admin_menu()
        elif user_role == 'client':
            render_client_menu()
        else:
            st.page_link("pages/0_🔐_Login.py", label="🔐 Se connecter", icon="🔐")
        
        st.markdown("---")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            <p>🛍️ <strong>Olist Analytics</strong></p>
            <p>Version 1.0.0</p>
            <p>© 2026 - Tous droits réservés</p>
        </div>
        """, unsafe_allow_html=True)

def render_admin_menu():
    """Menu de navigation pour les administrateurs"""
    
    st.page_link("app.py", label="🏠 Accueil", icon="🏠")
    st.page_link("pages/1_🏠_Dashboard_Admin.py", label="Dashboard KPI", icon="📊")
    
    st.markdown("#### 🤖 Modèles ML")
    st.page_link("pages/2_🚚_Prédiction_Livraison.py", label="Délais de livraison", icon="🚚")
    st.page_link("pages/3_💬_Analyse_Sentiment.py", label="Sentiment client", icon="💬")
    st.page_link("pages/4_📦_Prédiction_Commandes.py", label="Prévision commandes", icon="📦")
    
    st.markdown("#### 🛒 Vue Client")
    st.page_link("pages/5_🛒_Catalogue_Produits.py", label="Catalogue produits", icon="🛒")
    
    st.markdown("#### ⚙️ Gestion")
    st.page_link("pages/6_⚙️_Gestion_Modèles.py", label="Gestion des modèles", icon="⚙️")

def render_client_menu():
    """Menu de navigation pour les clients"""
    
    st.page_link("pages/5_🛒_Catalogue_Produits.py", label=get_text('search'), icon="🔍")
