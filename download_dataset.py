"""
download_dataset.py (v2)
========================
iNaturalist API'den çok sayfalı istek ile her tür için
en az 150 yüksek kaliteli görsel indirir.
"""

import os
import time
import requests

SPECIES = {
    "caretta_caretta":        "Caretta caretta",
    "chelonia_mydas":         "Chelonia mydas",
    "eretmochelys_imbricata": "Eretmochelys imbricata",
}

DATASET_DIR       = "dataset"
TARGET_PER_SPECIES = 150   # Her tür için hedef görsel sayısı
PER_PAGE          = 50     # API başına max istek


def download_images():
    print("🐢 Gelişmiş Veri Seti İndirici v2 Başlıyor...\n")

    for folder_name, taxon_name in SPECIES.items():
        save_dir = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # Mevcut dosyaları say, sadece eksik olanları indir
        existing = [
            f for f in os.listdir(save_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        already = len(existing)
        need    = max(0, TARGET_PER_SPECIES - already)

        if need == 0:
            print(f"⏭️  {folder_name}: Zaten {already} görsel var, atlanıyor.\n")
            continue

        print(f"📥 {taxon_name}: {already} mevcut → {need} daha indirilecek...")

        count     = 0
        page      = 1
        max_pages = (TARGET_PER_SPECIES // PER_PAGE) + 3  # biraz pay bırak

        while count < need and page <= max_pages:
            url = (
                f"https://api.inaturalist.org/v1/observations"
                f"?taxon_name={taxon_name}"
                f"&has[]=photos"
                f"&quality_grade=research"    # sadece bilim insanı onaylı
                f"&per_page={PER_PAGE}"
                f"&page={page}"
                f"&order=created_at&order_by=desc"  # farklı sayfalar = farklı görseller
            )

            try:
                resp = requests.get(url, timeout=15)
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for obs in results:
                    if count >= need:
                        break
                    if not obs.get("photos"):
                        continue

                    # Tüm fotoğrafları al (gözlem başına birden fazla olabilir)
                    for photo in obs["photos"][:1]:   # ilk fotoğraf yeterli
                        photo_url = photo["url"].replace("square", "medium")
                        try:
                            img_data = requests.get(photo_url, timeout=10).content
                            idx      = already + count + 1
                            file_path = os.path.join(
                                save_dir, f"{folder_name}_{idx:04d}.jpg"
                            )
                            with open(file_path, "wb") as f:
                                f.write(img_data)
                            count += 1
                        except Exception as e:
                            pass  # Tek bir görsel hatasında devam et
                        time.sleep(0.05)

                page += 1
                time.sleep(0.3)  # API limitini aşmamak için

            except Exception as e:
                print(f"  ⚠️  API hatası (sayfa {page}): {e}")
                break

        total_now = already + count
        print(f"  ✅ {folder_name}: +{count} yeni → Toplam {total_now} görsel\n")

    # Özet
    print("\n📊 Veri Seti Özeti:")
    for folder_name in SPECIES:
        save_dir = os.path.join(DATASET_DIR, folder_name)
        imgs = [f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        status = "✅" if len(imgs) >= 100 else "⚠️ "
        print(f"  {status} {folder_name}: {len(imgs)} görsel")


if __name__ == "__main__":
    download_images()
