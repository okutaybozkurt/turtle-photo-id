"""
setup_photo_id.py
=================
Kaggle'dan indirilen SeaTurtleIDHeads veri setindeki kaplumbağaları
"Photo-ID" birey veritabanımıza kaydeder.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db import get_connection
from src.preprocessing import preprocess_for_model
from src.model import get_embedding_model, extract_embedding
from src.matcher import save_embedding

KAGGLE_DIR = "dataset_kaggle/images"
# Artık sınır yok, klasördeki tüm kaplumbağaları alıyoruz.
IMAGES_PER_TURTLE = 2 # İşlemi hızlandırmak için her bireyin en iyi 2 fotoğrafı alınır

def reset_database():
    """Veritabanındaki eski Photo-ID kayıtlarını temizler."""
    conn = get_connection()
    if not conn:
        return
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM photo_embeddings")
        cursor.execute("DELETE FROM turtles")
        conn.commit()
        print("Veritabanı temizlendi, baştan kuruluyor...")
    except Exception as e:
        print("Veritabanı temizleme hatası:", e)
    finally:
        cursor.close()
        conn.close()

def ensure_unknown_species():
    """Bireylerin türünü bilmediğimiz için veritabanına 'Bilinmeyen' türü ekler."""
    conn = get_connection()
    if not conn:
        return 1
    cursor = conn.cursor()
    cursor.execute("""
        INSERT IGNORE INTO species (code, name_sci, name_tr)
        VALUES ('unknown', 'Chelonioidea', 'Bilinmeyen Deniz Kaplumbağası')
    """)
    conn.commit()
    
    cursor.execute("SELECT id FROM species WHERE code='unknown'")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else 1

def register_turtle(internal_code: str, species_id: int) -> int:
    """Yeni bireyi turtles tablosuna ekler ve ID'sini döner."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM turtles WHERE internal_code = %s", (internal_code,))
    row = cursor.fetchone()
    if row:
        return row[0]
        
    cursor.execute("""
        INSERT INTO turtles (internal_code, species_id, notes)
        VALUES (%s, %s, %s)
    """, (internal_code, species_id, "Kaggle SeaTurtleID dataset"))
    turtle_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return turtle_id

def main():
    print("==================================================")
    print("Turtle Photo-ID Veritabanı Kurulumu Başlıyor...")
    print("==================================================")
    
    if not os.path.exists(KAGGLE_DIR):
        print(f"Hata: {KAGGLE_DIR} bulunamadı.")
        sys.exit(1)
        
    reset_database()
    species_id = ensure_unknown_species()
    model = get_embedding_model()
    
    turtle_folders = [f for f in os.listdir(KAGGLE_DIR) if os.path.isdir(os.path.join(KAGGLE_DIR, f))]
    turtle_folders.sort()
    
    print(f"Toplam {len(turtle_folders)} farklı kaplumbağa bulundu. Tümü işleniyor...")
    
    processed_count = 0
    for t_folder in turtle_folders:
        internal_code = t_folder # 't106', 't281' vs.
        folder_path = os.path.join(KAGGLE_DIR, t_folder)
        images = glob.glob(os.path.join(folder_path, "*.JPG")) + glob.glob(os.path.join(folder_path, "*.jpg"))
        
        if not images:
            continue
            
        turtle_id = register_turtle(internal_code, species_id)
        
        saved_imgs = 0
        for img_path in images[:IMAGES_PER_TURTLE]:
            prep_img = preprocess_for_model(img_path)
            if prep_img is None:
                continue
                
            embedding = extract_embedding(model, prep_img)
            success = save_embedding(turtle_id, img_path, embedding)
            if success:
                saved_imgs += 1
                
        if saved_imgs > 0:
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"... {processed_count} kaplumbağa sisteme eklendi ...")
            
    print("==================================================")
    print(f"İşlem Tamamlandı! Toplam {processed_count} kaplumbağa Photo-ID sistemine kaydedildi.")

if __name__ == "__main__":
    main()
