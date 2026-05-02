"""
src/db.py
=========
MySQL veritabanı katmanı.

SOLID Prensipleri:
- SRP: Her fonksiyon tek bir sorumluluğa sahip.
- OCP: Bağlantı yönetimi context manager ile genişletilebilir.
- DIP: Bağlantı detayları os.getenv ile dışarıdan enjekte edilir.
"""

import os
from contextlib import contextmanager
from typing import Optional

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()


# ── Bağlantı Fabrikası ────────────────────────────────────────────────────

def _build_config() -> dict:
    """Ortam değişkenlerinden DB yapılandırması oluşturur (DIP)."""
    return {
        "host":     os.getenv("DB_HOST", "localhost"),
        "user":     os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "turtle_photoid"),
        "charset":  "utf8mb4",
    }


def get_connection() -> Optional[mysql.connector.MySQLConnection]:
    """Tek kullanımlık MySQL bağlantısı döndürür."""
    try:
        return mysql.connector.connect(**_build_config())
    except Error as exc:
        print(f"[DB] Bağlantı hatası: {exc}")
        return None


@contextmanager
def managed_connection():
    """
    Context manager: bağlantı + cursor açar, bitince her zaman kapatır.
    Kullanım:
        with managed_connection() as (conn, cursor):
            cursor.execute(...)
    """
    conn = get_connection()
    if conn is None:
        raise RuntimeError("[DB] Veritabanına bağlanılamadı.")
    cursor = conn.cursor(dictionary=True)
    try:
        yield conn, cursor
        conn.commit()
    except Error as exc:
        conn.rollback()
        raise exc
    finally:
        cursor.close()
        conn.close()


# ── Şema Kurulum ──────────────────────────────────────────────────────────

_SCHEMA_SQL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS species (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        code        VARCHAR(50)  NOT NULL UNIQUE,
        name_sci    VARCHAR(100) NOT NULL,
        name_tr     VARCHAR(100),
        description TEXT,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS turtles (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        internal_code   VARCHAR(50) UNIQUE,
        species_id      INT NOT NULL,
        sex             ENUM('Dişi','Erkek','Bilinmiyor') DEFAULT 'Bilinmiyor',
        first_seen_date DATE,
        notes           TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (species_id) REFERENCES species(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS observations (
        id                INT AUTO_INCREMENT PRIMARY KEY,
        turtle_id         INT,
        species_id        INT NOT NULL,
        image_path        VARCHAR(255),
        predicted_species VARCHAR(100),
        confidence_score  FLOAT,
        location_name     VARCHAR(150),
        latitude          DECIMAL(9,6),
        longitude         DECIMAL(9,6),
        observed_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
        observer_notes    TEXT,
        FOREIGN KEY (turtle_id)  REFERENCES turtles(id),
        FOREIGN KEY (species_id) REFERENCES species(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS photo_embeddings (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        turtle_id      INT NOT NULL,
        observation_id INT,
        image_path     VARCHAR(255) NOT NULL,
        embedding      LONGBLOB NOT NULL,
        model_version  VARCHAR(50) DEFAULT 'efficientnetb0_v2',
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (turtle_id)      REFERENCES turtles(id),
        FOREIGN KEY (observation_id) REFERENCES observations(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]

_SEED_SPECIES = [
    ("caretta_caretta",        "Caretta caretta",        "Akdeniz Caretta'sı"),
    ("chelonia_mydas",         "Chelonia mydas",         "Yeşil Deniz Kaplumbağası"),
    ("eretmochelys_imbricata", "Eretmochelys imbricata", "Hawksbill Kaplumbağası"),
    ("unknown",                "Chelonioidea",           "Bilinmeyen Deniz Kaplumbağası"),
]


def init_database() -> bool:
    """Şemayı ve başlangıç verilerini oluşturur."""
    conn = get_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        for sql in _SCHEMA_SQL:
            cursor.execute(sql)

        cursor.executemany(
            "INSERT IGNORE INTO species (code, name_sci, name_tr) VALUES (%s, %s, %s)",
            _SEED_SPECIES,
        )
        conn.commit()
        print("[DB] Şema hazır.")
        return True
    except Error as exc:
        print(f"[DB] Şema hatası: {exc}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


# ── CRUD İşlemleri ────────────────────────────────────────────────────────

def get_species_id(code: str) -> Optional[int]:
    """Tür kodundan species.id döndürür (SRP)."""
    conn = get_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM species WHERE code = %s", (code,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()
        conn.close()


def save_observation(
    species_id: int,
    predicted_species: str,
    confidence: float,
    image_path: Optional[str] = None,
    location: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[int]:
    """Gözlem kaydeder; başarıda yeni satır ID'si, başarısızda None döner."""
    conn = get_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO observations
                (species_id, predicted_species, confidence_score,
                 image_path, location_name, observer_notes)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (species_id, predicted_species, confidence, image_path, location, notes),
        )
        conn.commit()
        return cursor.lastrowid
    except Error as exc:
        print(f"[DB] Gözlem kayıt hatası: {exc}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()


def get_all_observations() -> list[dict]:
    """Tüm gözlemleri tarih sırasıyla döndürür."""
    conn = get_connection()
    if not conn:
        return []
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT o.id, o.predicted_species, o.confidence_score,
                   o.location_name, o.observed_at, o.observer_notes,
                   s.name_sci, s.name_tr
            FROM observations o
            JOIN species s ON o.species_id = s.id
            ORDER BY o.observed_at DESC
            """
        )
        return cursor.fetchall()
    except Error as exc:
        print(f"[DB] Sorgulama hatası: {exc}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_observation_stats() -> dict:
    """Dashboard için özet istatistik döndürür."""
    conn = get_connection()
    if not conn:
        return {}
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT COUNT(*) AS total FROM observations")
        total = cursor.fetchone()["total"]

        cursor.execute(
            """
            SELECT s.name_tr, COUNT(*) AS cnt
            FROM observations o
            JOIN species s ON o.species_id = s.id
            GROUP BY s.name_tr
            ORDER BY cnt DESC
            """
        )
        by_species = cursor.fetchall()

        cursor.execute("SELECT COUNT(DISTINCT id) AS total FROM turtles")
        turtle_count = cursor.fetchone()["total"]

        return {
            "total_observations": total,
            "by_species":         by_species,
            "registered_turtles": turtle_count,
        }
    except Error as exc:
        print(f"[DB] İstatistik hatası: {exc}")
        return {}
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    init_database()
