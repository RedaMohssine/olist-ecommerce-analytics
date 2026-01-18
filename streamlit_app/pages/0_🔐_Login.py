"""
Page de connexion
"""

import streamlit as st
from database.auth_db import authenticate_user, update_last_login, log_session
from components.style import load_global_styles

# Configuration de la page
st.set_page_config(
    page_title="Connexion - Olist Analytics",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS global
load_global_styles()


# CSS personnalisé pour la page de login
st.markdown("""
<style>
    /* Masquer la sidebar sur la page de login */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Centrer le contenu */
    .main .block-container {
        max-width: 500px;
        padding-top: 5rem;
    }
    
    /* Carte de login */
    .login-card {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Logo et titre */
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .login-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        color: #64748b;
        font-size: 1rem;
    }
    /* Style des inputs */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #009739;
        box-shadow: 0 0 0 3px rgba(0, 151, 57, 0.1);
    }
    
    /* Bouton de connexion */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #009739 0%, #00b368 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0, 151, 57, 0.3);
    }
    
    /* Info box */
    .info-box {
        background: #f1f5f9;
        border-left: 4px solid #009739;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 2rem;
    }
    
    .info-box h4 {
        color: #009739;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .info-box p {
        margin: 0.25rem 0;
        font-size: 0.85rem;
        color: #475569;
    }
    
    /* Footer */
    .login-footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="login-card">
    <div class="login-header">
        <div class="login-logo">🚀</div>
        <h1 class="login-title">Olist Analytics</h1>
        <p class="login-subtitle">Plateforme d'analyse e-commerce</p>
    </div>
""", unsafe_allow_html=True)

# Vérifier si déjà connecté
if st.session_state.get('authenticated'):
    st.success("✅ Vous êtes déjà connecté !")
    st.info(f"**Utilisateur :** {st.session_state.get('user_full_name')}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Aller à la page", width='stretch'):
            if st.session_state.get('user_role') == 'admin':
                st.switch_page("pages/1_🏠_Dashboard_Admin.py")
            else:
                st.switch_page("pages/5_🛒_Catalogue_Produits.py")
    
    with col2:
        if st.button("🚪 Se déconnecter", width='stretch'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

else:
    # Formulaire de connexion
    with st.form("login_form"):
        username = st.text_input(
            "👤 Nom d'utilisateur",
            placeholder="Entrez votre nom d'utilisateur"
        )
        
        password = st.text_input(
            "🔒 Mot de passe",
            type="password",
            placeholder="Entrez votre mot de passe"
        )
        
        submit = st.form_submit_button("🚀 Se connecter")
        
        if submit:
            if username and password:
                # Tentative d'authentification
                user = authenticate_user(username, password)
                
                if user:
                    # Connexion réussie
                    st.session_state['authenticated'] = True
                    st.session_state['user_id'] = user['id']
                    st.session_state['username'] = user['username']
                    st.session_state['user_full_name'] = user['full_name']
                    st.session_state['user_email'] = user['email']
                    st.session_state['user_role'] = user['role']
                    
                    # Mettre à jour la dernière connexion
                    update_last_login(user['id'])
                    log_session(user['id'])
                    
                    st.success(f"✅ Bienvenue {user['full_name']} !")
                    st.balloons()
                    
                    # Redirection selon le rôle
                    if user['role'] == 'admin':
                        st.info("🔄 Redirection vers le Dashboard Admin...")
                        st.switch_page("pages/1_🏠_Dashboard_Admin.py")
                    else:
                        st.info("🔄 Redirection vers le Catalogue Produits...")
                        st.switch_page("pages/5_🛒_Catalogue_Produits.py")
                else:
                    st.error("❌ Nom d'utilisateur ou mot de passe incorrect")
            else:
                st.warning("⚠️ Veuillez remplir tous les champs")
    
    # Informations de connexion
    st.markdown("""
    <div class="info-box">
        <h4>📋 Comptes de démonstration</h4>
        <p><strong>Administrateur :</strong> admin / admin123</p>
        <p><strong>Client :</strong> client / client123</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="login-footer">
        <p>© 2026 Olist Analytics - Tous droits réservés</p>
        <p>Plateforme sécurisée d'analyse de données e-commerce</p>
    </div>
</div>
""", unsafe_allow_html=True)
