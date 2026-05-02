"""
src/matcher.py
===============
Photo-ID benzerlik modülü.
Pretrained MobileNetV2 ile embedding çıkarır ve
MySQL'deki kayıtlarla cosine similarity karşılaştırması yapar.
"""

import io
import struct
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.db import get_connection
from src.model import get_embedding_model, extract_embedding


# ── Yardımcı: Numpy ↔ Bytes ───────────────────────────────────────────────
def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Numpy float32 dizisini MySQL BLOB için bytes'a çevirir."""
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(raw: bytes) -> np.ndarray:
    """MySQL BLOB'dan numpy dizisine geri çevirir."""
    return np.frombuffer(raw, dtype=np.float32)


# ── Embedding Kaydetme ─────────────────────────────────────────────────────
def save_embedding(turtle_id: int, image_path: str,
                   embedding: np.ndarray,
                   observation_id: int = None) -> bool:
    """
    Bir kaplumbağaya ait embedding vektörünü MySQL'e kaydeder.
    """
    conn = get_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO photo_embeddings
                (turtle_id, observation_id, image_path, embedding)
            VALUES (%s, %s, %s, %s)
        """, (turtle_id, observation_id, image_path,
              embedding_to_bytes(embedding)))
        conn.commit()
        return True
    except Exception as e:
        print(f"[Matcher] Embedding kayıt hatası: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


# ── Benzerlik Arama ────────────────────────────────────────────────────────
def find_similar(query_embedding: np.ndarray,
                 top_k: int = 3) -> list[dict]:
    """
    Sorgu embedding'ini MySQL'deki tüm kayıtlarla karşılaştırır.
    En benzer top_k kaydı döndürür.

    Returns:
        [
            {
                "turtle_id":    1,
                "internal_code": "CC-001",
                "image_path":   "...",
                "similarity":   0.95
            },
            ...
        ]
    """
    conn = get_connection()
    if not conn:
        return []

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT pe.id, pe.turtle_id, pe.image_path, pe.embedding,
                   t.internal_code, s.name_sci, s.name_tr
            FROM photo_embeddings pe
            JOIN turtles t  ON pe.turtle_id = t.id
            JOIN species  s ON t.species_id  = s.id
        """)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"[Matcher] Sorgulama hatası: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

    if not rows:
        return []

    # Embedding'leri numpy matrisine dönüştür
    db_embeddings = np.stack(
        [bytes_to_embedding(row["embedding"]) for row in rows]
    )

    # Cosine similarity hesapla — (1, N) matrisinden N skorlu dizi
    query = query_embedding.reshape(1, -1)
    scores = cosine_similarity(query, db_embeddings)[0]

    # En iyi top_k indeksi bul
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = rows[idx]
        results.append({
            "turtle_id":     row["turtle_id"],
            "internal_code": row["internal_code"],
            "image_path":    row["image_path"],
            "species_sci":   row["name_sci"],
            "species_tr":    row["name_tr"],
            "similarity":    float(scores[idx]),
        })

    return results


# ── Tam Pipeline ───────────────────────────────────────────────────────────
def match_photo(preprocessed_img: np.ndarray,
                top_k: int = 3,
                embedding_model=None) -> list[dict]:
    """
    Ön işlenmiş görüntüden embedding çıkarır ve en benzer bireyleri döndürür.

    Args:
        preprocessed_img: (224, 224, 3) float32 numpy dizisi.
        top_k: Döndürülecek sonuç sayısı.
        embedding_model: Cache'lenmiş Keras modeli.

    Returns:
        find_similar() çıktısı.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()
    query_emb = extract_embedding(embedding_model, preprocessed_img)
    return find_similar(query_emb, top_k=top_k)
