# Olist E-Commerce Analytics Platform

Une plateforme d'analyse e-commerce intelligente propulsée par l'IA et le Machine Learning, construite avec Streamlit et Python. Cette application offre des fonctionnalités avancées d'analyse de données, de prédiction et de visualisation pour optimiser les performances d'une plateforme e-commerce.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)

## Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Modèles ML](#-modèles-ml)
- [Datasets](#-datasets)
- [Authentification](#-authentification)
- [Contribution](#-contribution)

## Fonctionnalités

### Authentification & Rôles
- Système d'authentification sécurisé avec SQLite
- Deux rôles utilisateur : **Admin** et **Client**
- Interface adaptée selon les permissions

### Dashboard Administrateur
- Visualisation des KPIs en temps réel (revenus, commandes, satisfaction)
- Graphiques interactifs avec Plotly
- Analyse de tendances et insights business
- Exportation de rapports CSV

### Prédiction de Délais de Livraison
- Modèle XGBoost avec **MAE de 3.42 jours**
- Prédiction individuelle ou par lot (CSV)
- Prise en compte de multiples facteurs : distance, poids, état client/vendeur
- Visualisation des délais par région

### Analyse de Sentiment
- Classification automatique des avis clients (positif/neutre/négatif)
- Modèle ML avec **accuracy de 80.5%**
- Analyse par lot et analyse par vendeur
- Identification des vendeurs top/flop selon sentiment

### Prédiction de Commandes
- Prévision du nombre de commandes par produit/mois avec XGBoost
- **R² de 0.88** sur données historiques
- Planification du stock automatique
- Alertes de rupture de stock

### Catalogue Produits
- Système de recommandation KNN (32 953 produits)
- Filtrage avancé par catégorie, prix, stock
- Affichage de statistiques produit en temps réel
- Estimation de livraison intégrée

### Gestion de Modèles ML
- Upload/versioning de modèles personnalisés
- Historique des modèles avec restauration
- Configuration des hyperparamètres
- Tests de modèle en temps réel

## Technologies

### Backend & Data Science
- **Python 3.9+**
- **Pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **Scikit-learn** - Preprocessing et KNN
- **XGBoost** - Modèles de régression/classification
- **NLTK** - NLP pour analyse de sentiment

### Frontend & Visualisation
- **Streamlit** - Framework web interactif
- **Plotly** - Graphiques interactifs
- **HTML/CSS** - Styling personnalisé

### Base de données
- **SQLite** - Authentification et gestion utilisateurs

### Machine Learning
- **Joblib/Pickle** - Sérialisation de modèles
- **TF-IDF** - Vectorisation de texte
- **Pipeline Sklearn** - Workflows ML

## Installation

### Prérequis
- Python 3.9 ou supérieur
- pip (gestionnaire de packages Python)

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/RedaMohssine/olist-analytics-platform.git
cd olist-analytics-platform
```

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
cd streamlit_app
pip install -r requirements.txt
```

4. **Télécharger les ressources NLTK** (pour analyse de sentiment)
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"
```

5. **Vérifier les données**
Assurez-vous que le dossier `Data/` contient les datasets Olist CSV :
```
Data/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_orders_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

## Utilisation

### Lancer l'application

```bash
cd streamlit_app
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

### Comptes de démonstration

| Rôle | Identifiant | Mot de passe | Accès |
|------|-------------|--------------|-------|
| Administrateur | `admin` | `admin123` | Toutes les fonctionnalités |
| Client | `client` | `client123` | Catalogue produits uniquement |

### Navigation

**Interface Admin :**
1. Dashboard KPI → Statistiques globales
2. Prédiction Livraison → Estimation de délais
3. Analyse Sentiment → Classification d'avis
4. Prédiction Commandes → Prévision de ventes
5. Catalogue Produits → Vue client
6. Gestion Modèles → Upload/config ML

**Interface Client :**
- Accès au catalogue produits avec recherche et filtres
- Recommandations personnalisées
- Estimation de livraison

## Structure du projet

```
ANALYTICS/
├── Data/                           # Datasets CSV Olist (99,441 commandes)
├── notebooks/                      # Jupyter notebooks (développement ML)
│   ├── analytics_shipping_time.ipynb
│   └── Sentimental_analysisv2.ipynb
├── streamlit_app/                 # Application principale
│   ├── app.py                     # Point d'entrée
│   ├── pages/                     # Pages Streamlit
│   │   ├── 0_🔐_Login.py
│   │   ├── 1_🏠_Dashboard_Admin.py
│   │   ├── 2_🚚_Prédiction_Livraison.py
│   │   ├── 3_💬_Analyse_Sentiment.py
│   │   ├── 4_📦_Prédiction_Commandes.py
│   │   ├── 5_🛒_Catalogue_Produits.py
│   │   └── 6_⚙️_Gestion_Modèles.py
│   ├── components/                # Composants réutilisables
│   │   ├── auth.py               # Authentification
│   │   ├── charts.py             # Graphiques Plotly
│   │   ├── product_card.py       # Cartes produits
│   │   ├── sidebar.py            # Menu latéral
│   │   ├── topbar.py             # Barre de connexion
│   │   └── translations.py       # Multilingue (FR/EN)
│   ├── utils/                     # Utilitaires
│   │   ├── data_loader.py        # Chargement données
│   │   ├── model_manager.py      # Gestion modèles ML
│   │   ├── orders_forecast.py    # Prédiction commandes
│   │   ├── recommendation_engine.py  # KNN recommandations
│   │   └── shipping_forecast.py  # Prédiction livraison
│   ├── models/                    # Modèles ML sauvegardés
│   │   ├── orders_forecast/      # XGBoost commandes
│   │   ├── shipping_forecast/    # XGBoost livraison
│   │   ├── sentiment/            # Sentiment + TF-IDF
│   │   └── recommendation/       # KNN 32k produits
│   ├── database/                  # Base de données auth
│   │   └── auth_db.py
│   ├── assets/                    # CSS/images
│   │   └── styles.css
│   ├── config/                    # Configurations
│   │   ├── models_config.json
│   │   └── settings.py
│   ├── .streamlit/               # Config Streamlit
│   │   └── config.toml
│   └── requirements.txt          # Dépendances Python
└── README.md                      # Ce fichier
```

## Modèles ML

### 1. Prédiction de Livraison (XGBoost Regressor)
- **Métriques** : MAE 3.42 jours | R² 0.85
- **Features** : 20 variables (poids, distance, état, catégorie, etc.)
- **Entraînement** : 96,478 commandes livrées
- **Fichier** : `models/shipping_forecast/xgboost_pipeline.pkl`

### 2. Analyse de Sentiment (Logistic Regression + TF-IDF)
- **Métriques** : Accuracy 80.5% | F1-weighted 0.81
- **Classes** : Positif (4-5★) / Neutre (3★) / Négatif (1-2★)
- **Corpus** : 99,224 avis clients en portugais
- **Fichiers** : 
  - `models/sentiment/sentiment_model.pkl`
  - `models/sentiment/tfidf_vectorizer.pkl`

### 3. Prédiction de Commandes (XGBoost Regressor)
- **Métriques** : R² 0.88 | MAE 2.3
- **Prédiction** : Nombre de commandes par produit/mois
- **Features** : Historique ventes, tendances, catégorie
- **Fichier** : `models/orders_forecast/xgboost_model.pkl`

### 4. Recommandations Produits (KNN)
- **Algorithme** : K-Nearest Neighbors (K=20)
- **Distance** : Cosinus (features numériques normalisées)
- **Corpus** : 32,953 produits
- **Fichier** : `models/recommendation/knn_model.pkl`

## Datasets

Le projet utilise le **Brazilian E-Commerce Public Dataset by Olist** (Kaggle).

### Statistiques
- **99,441** commandes (2016-2018)
- **32,953** produits uniques
- **3,095** vendeurs
- **99,441** clients
- **73 catégories** de produits

### Fichiers CSV
1. `olist_orders_dataset.csv` - Commandes principales
2. `olist_order_items_dataset.csv` - Items de commande
3. `olist_products_dataset.csv` - Catalogue produits
4. `olist_customers_dataset.csv` - Clients
5. `olist_sellers_dataset.csv` - Vendeurs
6. `olist_order_reviews_dataset.csv` - Avis clients
7. `olist_order_payments_dataset.csv` - Paiements
8. `olist_geolocation_dataset.csv` - Coordonnées GPS
9. `product_category_name_translation.csv` - Traduction catégories

**Source** : [Olist Dataset sur Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Authentification

### Système d'authentification
- Base de données SQLite (`database/auth.db`)
- Hashage de mots de passe avec `hashlib.sha256`
- Session Streamlit pour persistance
- Redirection automatique selon rôle

### Créer un nouvel utilisateur

Modifier `database/auth_db.py` :
```python
def create_default_users():
    # Ajouter votre utilisateur
    create_user("nouveau_user", "mot_de_passe", "Nom Complet", "admin")
```

## Personnalisation

### Changer les couleurs
Modifier `.streamlit/config.toml` :
```toml
[theme]
primaryColor = "#009739"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Modifier les seuils de sentiment
Dans `config/models_config.json` :
```json
"model_settings": {
  "sentiment": {
    "threshold_positive": 0.61,
    "threshold_negative": 0.39
  }
}
```

## Scripts Jupyter Notebooks

Les notebooks dans `notebooks/` contiennent :
- **analytics_shipping_time.ipynb** : Entraînement du modèle XGBoost de livraison
- **Sentimental_analysisv2.ipynb** : Entraînement du modèle de sentiment

Ces notebooks sont fournis pour reproduire l'entraînement des modèles.

## Troubleshooting

### Erreur : "Modèle non disponible"
- Vérifier que les fichiers `.pkl` sont présents dans `models/`
- Réentraîner les modèles via les notebooks si nécessaire

### Erreur : "Données non disponibles"
- Vérifier que tous les CSV sont dans le dossier `Data/`
- Vérifier les noms de fichiers (sensibles à la casse)

### Erreur NLTK
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"
```

### Port déjà utilisé
```bash
streamlit run app.py --server.port 8502
```

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## Auteur

**Reda** - Développeur Data Scientist

## Remerciements

- **Olist** pour le dataset public
- **Streamlit** pour le framework
- **XGBoost/Scikit-learn** pour les outils ML
- Communauté Kaggle pour les inspirations

## Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.
ou sur mon email: mohamed.mohssine@emines.um6p.ma
