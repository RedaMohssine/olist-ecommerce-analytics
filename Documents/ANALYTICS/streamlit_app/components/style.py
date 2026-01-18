"""
Chargement des styles globaux
"""

from pathlib import Path
import streamlit as st


def load_global_styles():
    """Injecte le CSS global depuis assets/styles.css"""
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
