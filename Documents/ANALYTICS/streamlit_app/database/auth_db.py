"""
Gestion de la base de données d'authentification
"""

import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "users.db"

def hash_password(password: str) -> str:
    """Hash un mot de passe avec SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_database():
    """Initialise la base de données avec la table users"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Créer la table users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    # Créer la table sessions (pour tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            logout_time TIMESTAMP,
            ip_address TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()

def create_default_users():
    """Crée les utilisateurs par défaut si ils n'existent pas"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Vérifier si des utilisateurs existent
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Créer compte admin
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email, role)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "admin",
            hash_password("admin123"),
            "Administrateur",
            "admin@olist.com",
            "admin"
        ))
        
        # Créer compte client
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email, role)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "client",
            hash_password("client123"),
            "Client Demo",
            "client@olist.com",
            "client"
        ))
        
        conn.commit()
        print("✅ Utilisateurs par défaut créés :")
        print("   Admin - username: admin, password: admin123")
        print("   Client - username: client, password: client123")
    
    conn.close()

def authenticate_user(username: str, password: str) -> dict:
    """
    Authentifie un utilisateur
    
    Returns:
        dict avec les infos user si authentifié, None sinon
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    
    cursor.execute("""
        SELECT id, username, full_name, email, role, is_active
        FROM users
        WHERE username = ? AND password_hash = ?
    """, (username, password_hash))
    
    result = cursor.fetchone()
    conn.close()
    
    if result and result[5]:  # is_active
        return {
            'id': result[0],
            'username': result[1],
            'full_name': result[2],
            'email': result[3],
            'role': result[4]
        }
    
    return None

def update_last_login(user_id: int):
    """Met à jour la dernière connexion d'un utilisateur"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE users
        SET last_login = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (user_id,))
    
    conn.commit()
    conn.close()

def log_session(user_id: int, ip_address: str = None):
    """Enregistre une session de connexion"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO sessions (user_id, ip_address)
        VALUES (?, ?)
    """, (user_id, ip_address))
    
    conn.commit()
    conn.close()

def get_all_users():
    """Récupère tous les utilisateurs (pour admin)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, full_name, email, role, created_at, last_login, is_active
        FROM users
        ORDER BY created_at DESC
    """)
    
    users = []
    for row in cursor.fetchall():
        users.append({
            'id': row[0],
            'username': row[1],
            'full_name': row[2],
            'email': row[3],
            'role': row[4],
            'created_at': row[5],
            'last_login': row[6],
            'is_active': bool(row[7])
        })
    
    conn.close()
    return users

# Initialiser la base au chargement du module
init_database()
create_default_users()
