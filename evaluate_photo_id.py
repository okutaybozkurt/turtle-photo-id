"""
evaluate_photo_id.py
====================
Photo-ID sisteminin başarımını ölçer.
TensorFlow kullanmadan sadece numpy/sklearn ile hesaplar — M1/M2 Mac uyumlu, hızlı.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI açmadan PNG kaydet
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db import get_connection
from src.matcher import bytes_to_embedding

KAGGLE_DIR = "dataset_kaggle/images"


# ── Yardımcılar ───────────────────────────────────────────────────────────

def preprocess_np(img_path: str, size: tuple = (224, 224)) -> np.ndarray | None:
    """
    PIL ile görüntü yükler ve EfficientNet preprocess_input uygular.
    setup_photo_id.py ile aynı pipeline — tutarlılık kritik.
    """
    try:
        from PIL import Image
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img = Image.open(img_path).convert("RGB").resize(size)
        arr = np.array(img, dtype=np.float32)
        return preprocess_input(arr)
    except Exception as e:
        print(f"  [Hata] {img_path}: {e}")
        return None


def get_all_db_embeddings():
    """Veritabanındaki tüm embedding + metadata'yı tek sorguda çeker."""
    conn = get_connection()
    if not conn:
        return []
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT pe.embedding, t.internal_code
        FROM photo_embeddings pe
        JOIN turtles t ON pe.turtle_id = t.id
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def get_embedding_model():
    """
    EfficientNetB0 feature extractor — TF başlatma bağımsız fonksiyon.
    Yalnızca bir kez çağrılır.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, models

    base = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    emb_model = models.Sequential([base, layers.GlobalAveragePooling2D()], name="emb")
    return emb_model


# ── Ana Mantık ────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("Photo-ID Sistem Başarımı Değerlendiriliyor...")
    print("=" * 52)

    # Test seti: Her bireyin 3. fotoğrafı (ilk 2'si DB'de)
    test_queries = []
    for t_folder in sorted(os.listdir(KAGGLE_DIR)):
        folder_path = os.path.join(KAGGLE_DIR, t_folder)
        if not os.path.isdir(folder_path):
            continue
        imgs = sorted(
            glob.glob(os.path.join(folder_path, "*.JPG")) +
            glob.glob(os.path.join(folder_path, "*.jpg"))
        )
        if len(imgs) >= 3:
            test_queries.append({"true_id": t_folder, "image_path": imgs[2]})

    test_queries = test_queries[:100]
    total = len(test_queries)
    print(f"Test seti: {total} birey × 1 daha önce görülmemiş fotoğraf\n")

    # Veritabanı matrisini belleğe al
    print("Veritabanı embeddingleri yükleniyor...")
    db_rows = get_all_db_embeddings()
    if not db_rows:
        print("HATA: Veritabanında embedding yok.")
        return
    db_matrix  = np.stack([bytes_to_embedding(r["embedding"]) for r in db_rows])
    db_ids     = [r["internal_code"] for r in db_rows]
    print(f"  {len(db_ids)} referans vektör hazır.\n")

    # TF modelini yükle
    print("TensorFlow modeli yükleniyor (yalnızca 1 kez)...")
    model = get_embedding_model()
    print("  Model hazır.\n")

    top1_correct = 0
    top3_correct = 0

    for i, q in enumerate(test_queries):
        arr = preprocess_np(q["image_path"])
        if arr is None:
            continue

        batch  = arr[np.newaxis, ...]               # (1, 224, 224, 3)
        emb    = model(batch, training=False).numpy()[0]  # (1280,)
        scores = cosine_similarity(emb.reshape(1, -1), db_matrix)[0]

        top3_idx  = np.argsort(scores)[::-1][:3]
        top3_pred = [db_ids[j] for j in top3_idx]

        if top3_pred[0] == q["true_id"]:
            top1_correct += 1
        if q["true_id"] in top3_pred:
            top3_correct += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{total}]  Top-1: {top1_correct}/{i+1}")

    top1_acc = top1_correct / total * 100
    top3_acc = top3_correct / total * 100

    print()
    print("=" * 52)
    print("TEST SONUÇLARI")
    print(f"  Top-1 Accuracy : %{top1_acc:.1f}  ({top1_correct}/{total})")
    print(f"  Top-3 Accuracy : %{top3_acc:.1f}  ({top3_correct}/{total})")
    print("=" * 52)

    # Grafik
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Top-1 Doğruluk", "Top-3 Doğruluk"],
        [top1_acc, top3_acc],
        color=["#0f766e", "#14b8a6"],
        width=0.45,
        edgecolor="none",
    )
    for bar, val in zip(bars, [top1_acc, top3_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"%{val:.1f}",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Başarı Oranı (%)", fontsize=11)
    ax.set_title(
        "Photo-ID Birey Tanıma — Başarım Analizi\n"
        f"(n={total}, EfficientNetB0 + Cosine Similarity)",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = "evaluation_results.png"
    plt.savefig(out_path, dpi=180)
    print(f"\nGrafik kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
