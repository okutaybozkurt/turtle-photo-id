"""
src/utils.py
=============
Genel yardımcı fonksiyonlar.
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt


# ── Sınıf etiketleri ──────────────────────────────────────────────────────
CLASS_LABELS_TR = {
    "caretta_caretta":        "Caretta Caretta",
    "chelonia_mydas":         "Yeşil Deniz Kaplumbağası",
    "eretmochelys_imbricata": "Hawksbill Kaplumbağası",
}


def format_confidence(score: float) -> str:
    """0.93 → '%93.0' formatına çevirir."""
    return f"%{score * 100:.1f}"


def timestamp_str() -> str:
    """Şu anki zamanı 'YYYY-MM-DD HH:MM:SS' formatında döndürür."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Klasör yardımcıları ────────────────────────────────────────────────────
def ensure_dir(path: str) -> None:
    """Klasör yoksa oluşturur."""
    os.makedirs(path, exist_ok=True)


def count_images_per_class(dataset_dir: str) -> dict:
    """
    dataset/ klasöründeki her alt klasörde kaç görüntü olduğunu sayar.

    Returns:
        {"caretta_caretta": 42, "chelonia_mydas": 38, ...}
    """
    counts = {}
    if not os.path.isdir(dataset_dir):
        return counts

    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        counts[class_name] = len(images)

    return counts


# ── Eğitim grafikleri ──────────────────────────────────────────────────────
def plot_training_history(history, save_path: str = "models/training_history.png") -> None:
    """
    Keras History nesnesinden accuracy ve loss grafiklerini çizer ve kaydeder.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Doğruluk (Accuracy)
    axes[0].plot(history.history["accuracy"],     label="Eğitim", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Doğrulama", linewidth=2)
    axes[0].set_title("Model Doğruluğu (Accuracy)", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Doğruluk")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Kayıp (Loss)
    axes[1].plot(history.history["loss"],     label="Eğitim", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Doğrulama", linewidth=2)
    axes[1].set_title("Model Kaybı (Loss)", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Kayıp")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150)
    print(f"[Utils] Grafik kaydedildi: {save_path}")
    plt.close()


# ── Benzerlik sonuçlarını yazdır ───────────────────────────────────────────
def print_match_results(results: list[dict]) -> None:
    """Photo-ID match sonuçlarını terminale yazdırır."""
    if not results:
        print("Veritabanında eşleşen birey bulunamadı.")
        return

    print("\n📋 En Benzer Bireyler:")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        similarity_pct = f"{r['similarity'] * 100:.1f}%"
        print(f"  {i}. Birey Kodu : {r.get('internal_code', 'Bilinmiyor')}")
        print(f"     Tür         : {r.get('species_tr', '-')} ({r.get('species_sci', '-')})")
        print(f"     Benzerlik   : {similarity_pct}")
        print(f"     Görüntü     : {r.get('image_path', '-')}")
        print()
