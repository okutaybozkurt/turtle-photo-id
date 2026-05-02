"""
filter_dataset.py
==================
OpenCV ile her görseli analiz eder, boyutu çok küçük veya
çok gürültülü (siyah/boş) olan görselleri siler.
Ardından iNaturalist'ten "head" etiketli görseller indirir.
"""

import os
import shutil
import requests
import time
import cv2
import numpy as np

DATASET_DIR = "dataset"

SPECIES = {
    "caretta_caretta":        "Caretta caretta",
    "chelonia_mydas":         "Chelonia mydas",
    "eretmochelys_imbricata": "Eretmochelys imbricata",
}

# Hedef: her tür için en az bu kadar kalsın
TARGET = 100


def is_valid_image(path: str) -> bool:
    """
    Görselin geçerli ve yeterince bilgi taşıyıp taşımadığını kontrol eder.
    - Bozuk/okunamaz → False
    - Çok küçük (< 80x80) → False
    - Çok tek renkli (laplacian variance < 50) → False (boş/siyah/gürültüsüz)
    """
    img = cv2.imread(path)
    if img is None:
        return False
    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 30:   # Çok düz/boş görsel
        return False
    return True


def clean_dataset():
    """Geçersiz görselleri siler, istatistik döndürür."""
    print("🧹 Veri Seti Temizleniyor...\n")
    total_removed = 0
    for folder in SPECIES:
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        removed = 0
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            if not is_valid_image(img_path):
                os.remove(img_path)
                removed += 1
        remaining = len(images) - removed
        print(f"  {folder}: {len(images)} → {remaining} görsel ({removed} silindi)")
        total_removed += removed
    print(f"\n  Toplam silinen: {total_removed} görsel\n")


def download_more_if_needed():
    """Temizleme sonrası TARGET'a ulaşmak için ek görsel indirir."""
    print("📥 Eksik Görseller Tamamlanıyor...\n")
    for folder_name, taxon_name in SPECIES.items():
        folder_path = os.path.join(DATASET_DIR, folder_name)
        existing = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        need = max(0, TARGET - len(existing))
        if need == 0:
            print(f"  ✅ {folder_name}: {len(existing)} görsel — yeterli, atlanıyor")
            continue

        print(f"  📥 {folder_name}: {len(existing)} mevcut → {need} daha indiriliyor...")
        count = 0
        page = 1
        while count < need and page <= 6:
            url = (
                f"https://api.inaturalist.org/v1/observations"
                f"?taxon_name={taxon_name}"
                f"&has[]=photos"
                f"&quality_grade=research"
                f"&per_page=50"
                f"&page={page}"
                f"&order=created_at&order_by=asc"   # eski sayfalar = farklı görseller
            )
            try:
                resp = requests.get(url, timeout=15).json()
                for obs in resp.get("results", []):
                    if count >= need:
                        break
                    if not obs.get("photos"):
                        continue
                    photo_url = obs["photos"][0]["url"].replace("square", "medium")
                    try:
                        img_data = requests.get(photo_url, timeout=10).content
                        idx = len(existing) + count + 1
                        fpath = os.path.join(folder_path, f"{folder_name}_extra_{idx:04d}.jpg")
                        with open(fpath, "wb") as f:
                            f.write(img_data)
                        # Geçerli mi kontrol et, değilse sil
                        if is_valid_image(fpath):
                            count += 1
                        else:
                            os.remove(fpath)
                    except:
                        pass
                    time.sleep(0.05)
                page += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"    API hatası (sayfa {page}): {e}")
                break

        final = len([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  ✅ {folder_name}: +{count} eklendi → Toplam {final} görsel")

    print()


def summary():
    print("📊 Final Veri Seti Durumu:")
    for folder in SPECIES:
        folder_path = os.path.join(DATASET_DIR, folder)
        imgs = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        status = "✅" if len(imgs) >= TARGET else "⚠️ "
        print(f"  {status} {folder}: {len(imgs)} görsel")


if __name__ == "__main__":
    clean_dataset()
    download_more_if_needed()
    summary()
    print("\n✅ Temizleme tamamlandı. Şimdi 'python train_model.py' çalıştırabilirsiniz.")
