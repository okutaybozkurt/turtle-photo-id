"""
src/preprocessing.py
=====================
OpenCV tabanlı görüntü ön işleme fonksiyonları.
"""

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


# Model girdi boyutu (EfficientNetB0 için 224x224)
IMG_SIZE = (224, 224)


def load_image(source) -> np.ndarray | None:
    """
    Dosya yolu veya PIL Image'dan BGR numpy dizisi yükler.
    Streamlit'ten gelen UploadedFile da kabul edilir.
    """
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            print(f"[Preprocessing] Görüntü okunamadı: {source}")
        return img

    # Bytes veya file-like nesne (Streamlit UploadedFile)
    try:
        pil_img = Image.open(source).convert("RGB")
        img = np.array(pil_img)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[Preprocessing] Yükleme hatası: {e}")
        return None


def resize(img: np.ndarray, size: tuple = IMG_SIZE) -> np.ndarray:
    """Görseli hedef boyuta yeniden boyutlandırır."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) ile
    her renk kanalında kontrast artırma uygular.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def reduce_noise(img: np.ndarray) -> np.ndarray:
    """Gaussian blur ile gürültü azaltma."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def normalize(img: np.ndarray) -> np.ndarray:
    """
    BGR → RGB dönüşümü + EfficientNet preprocess_input.
    NOT: EfficientNet [0,255] aralığını bekler, /255 YAPILMAZ.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32)
    return preprocess_input(rgb)


def preprocess_for_model(source, apply_clahe: bool = False,
                         apply_denoise: bool = False) -> np.ndarray | None:
    """
    Eksiksiz ön işleme pipeline'ı:
      load → resize → (denoise) → EfficientNet preprocess_input

    Returns:
        (224, 224, 3) float32 numpy dizisi veya None (hata durumunda).
    """
    img = load_image(source)
    if img is None:
        return None

    img = resize(img)

    if apply_denoise:
        img = reduce_noise(img)

    if apply_clahe:
        img = enhance_contrast(img)

    return normalize(img)


def get_edge_map(img: np.ndarray) -> np.ndarray:
    """
    Canny kenar tespiti — görsel inceleme / debug amaçlıdır.
    Model pipeline'ında kullanılmaz.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, threshold1=50, threshold2=150)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Kullanım: python preprocessing.py <görüntü_yolu>")
        sys.exit(1)

    path = sys.argv[1]
    processed = preprocess_for_model(path)
    if processed is not None:
        print(f"✅ Ön işleme başarılı. Çıktı şekli: {processed.shape}")
    else:
        print("❌ Ön işleme başarısız.")
