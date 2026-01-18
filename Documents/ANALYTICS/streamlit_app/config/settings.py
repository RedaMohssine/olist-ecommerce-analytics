"""
Configuration centralisée de l'application
"""

import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "models_config.json"

def load_config():
    """Charge la configuration depuis le fichier JSON"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model_config(model_type):
    """Retourne la configuration d'un modèle spécifique"""
    config = load_config()
    return config['models'].get(model_type, {})

def get_data_paths():
    """Retourne les chemins vers les datasets"""
    config = load_config()
    return config['data_paths']

def get_app_settings():
    """Retourne les paramètres de l'application"""
    config = load_config()
    return config['app_settings']

# Constantes
PRODUCTS_PER_PAGE = 12
RECOMMENDATIONS_COUNT = 4
DEFAULT_LANGUAGE = 'fr'
CACHE_TTL = 3600

# Couleurs du thème Brésil
BRAZIL_COLORS = {
    'green': '#009739',
    'yellow': '#FEDD00',
    'blue': '#002776',
    'white': '#FFFFFF',
    'dark': '#1a1a1a'
}
