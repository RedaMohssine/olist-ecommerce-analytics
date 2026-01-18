"""
Barre d'action en haut à droite (Login/Logout)
"""

import streamlit as st


def render_topbar(key_prefix=""):
    cols = st.columns([11, 2])
    with cols[1]:
        if st.session_state.get('authenticated'):
            if st.button("🚪 Déconnexion", key=f"{key_prefix}top_logout", width=220):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.switch_page("app.py")
        else:
            if st.button("🔐 Se connecter", key=f"{key_prefix}top_login", width=220):
                st.switch_page("pages/0_🔐_Login.py")
