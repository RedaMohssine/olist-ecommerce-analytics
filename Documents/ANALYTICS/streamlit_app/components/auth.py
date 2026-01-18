"""
Composant d'authentification simple avec sélecteur Admin/Client
"""

import streamlit as st
from components.translations import get_text

def render_auth_selector():
    """Affiche le sélecteur de rôle utilisateur dans la sidebar"""
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔐 Profil Utilisateur")
        
        # Sélecteur de rôle
        role_options = {
            None: get_text("select_profile"),
            'admin': "👨‍💼 Administrateur",
            'client': "🛒 Client"
        }
        
        current_role = st.session_state.get('user_role', None)
        
        selected_role = st.selectbox(
            get_text("user_profile"),
            options=list(role_options.keys()),
            format_func=lambda x: role_options[x],
            index=list(role_options.keys()).index(current_role) if current_role in role_options else 0,
            key='role_selector'
        )
        
        # Mise à jour du rôle
        if selected_role != st.session_state.user_role:
            st.session_state.user_role = selected_role
            if selected_role is not None:
                st.success(f"✅ {get_text('logged_as')} {role_options[selected_role]}")
                st.rerun()
        
        # Bouton de déconnexion
        if st.session_state.user_role is not None:
            if st.button("🚪 " + get_text("logout"), width='stretch'):
                st.session_state.user_role = None
                st.rerun()
        
        st.markdown("---")

def check_user_role():
    """Retourne le rôle de l'utilisateur actuel"""
    return st.session_state.get('user_role', None)

def require_admin():
    """Redirige si l'utilisateur n'est pas admin"""
    if not st.session_state.get('authenticated'):
        st.switch_page("pages/0_🔐_Login.py")
        st.stop()
    if st.session_state.get('user_role') != 'admin':
        st.warning("⛔ Accès réservé aux administrateurs")
        st.switch_page("pages/5_🛒_Catalogue_Produits.py")
        st.stop()

def require_auth():
    """Redirige si l'utilisateur n'est pas connecté"""
    if not st.session_state.get('authenticated'):
        st.switch_page("pages/0_🔐_Login.py")
        st.stop()
