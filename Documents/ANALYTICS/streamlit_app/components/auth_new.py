"""
Gestion de l'authentification - Version complète
"""

import streamlit as st

def check_authentication():
    """Vérifie si l'utilisateur est authentifié"""
    if not st.session_state.get('authenticated'):
        st.error("🚫 Accès non autorisé. Veuillez vous connecter.")
        st.info("👉 Redirection vers la page de connexion...")
        
        # Bouton pour aller à la page de login
        if st.button("🔐 Se connecter", type="primary"):
            st.switch_page("pages/0_🔐_Login.py")
        
        st.stop()

def require_admin():
    """Vérifie que l'utilisateur est admin"""
    check_authentication()
    
    if st.session_state.get('user_role') != 'admin':
        st.error("🚫 Accès refusé. Cette page nécessite des droits administrateur.")
        st.info(f"👤 Connecté en tant que : **{st.session_state.get('user_full_name')}** ({st.session_state.get('user_role')})")
        st.stop()

def require_auth():
    """Vérifie que l'utilisateur est connecté (admin ou client)"""
    check_authentication()

def get_current_user():
    """Retourne les informations de l'utilisateur connecté"""
    if st.session_state.get('authenticated'):
        return {
            'id': st.session_state.get('user_id'),
            'username': st.session_state.get('username'),
            'full_name': st.session_state.get('user_full_name'),
            'email': st.session_state.get('user_email'),
            'role': st.session_state.get('user_role')
        }
    return None

def logout():
    """Déconnecte l'utilisateur"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def render_user_info():
    """Affiche les infos utilisateur dans la sidebar"""
    user = get_current_user()
    
    if user:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 Utilisateur")
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
                <div style='font-size: 1.2rem; font-weight: bold;'>{user['full_name']}</div>
                <div style='font-size: 0.9rem; opacity: 0.9;'>{user['email']}</div>
                <div style='margin-top: 0.5rem; padding: 0.3rem 0.6rem; background: rgba(255,255,255,0.2); 
                            border-radius: 5px; display: inline-block; font-size: 0.8rem;'>
                    {'🔑 Administrateur' if user['role'] == 'admin' else '👤 Client'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚪 Déconnexion", width='stretch', type="secondary"):
                logout()
