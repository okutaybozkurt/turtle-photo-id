"""
predict.py
===========
Tek bir görsel üzerinde tür tahmini yapar (komut satırından test için).

Kullanım:
    python predict.py <görüntü_yolu>

Örnek:
    python predict.py dataset/caretta_caretta/ornek.jpg
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_for_model
from src.model import load_model, predict
from src.utils import format_confidence

MODEL_PATH = "models/turtle_species_model.keras"


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <görüntü_yolu>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"❌ Dosya bulunamadı: {image_path}")
        sys.exit(1)

    if not os.path.isfile(MODEL_PATH):
        print(f"❌ Model bulunamadı: {MODEL_PATH}")
        print("   Önce 'python train_model.py' ile modeli eğitin.")
        sys.exit(1)

    print(f"\n🖼️  Görüntü: {image_path}")

    # Ön işleme
    processed = preprocess_for_model(image_path)
    if processed is None:
        print("❌ Görüntü ön işlenemedi.")
        sys.exit(1)

    # Model yükle ve tahmin yap
    model = load_model(MODEL_PATH)
    if model is None:
        sys.exit(1)

    result = predict(model, processed)

    # Sonuçları yazdır
    print("\n" + "=" * 45)
    print(f"  🐢 Tahmin Edilen Tür : {result['label_tr']}")
    print(f"  🔬 Bilimsel Adı       : {result['class']}")
    print(f"  📊 Güven Skoru        : {format_confidence(result['confidence'])}")
    print("=" * 45)

    print("\n  Tüm Skorlar:")
    for cls, score in result["all_scores"].items():
        bar = "█" * int(score * 30)
        print(f"  {cls:<30} {format_confidence(score):>7}  {bar}")

    print()


if __name__ == "__main__":
    main()
