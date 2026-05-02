"""
src/model.py
============
EfficientNetB0 transfer learning: tür sınıflandırma + embedding çıkarımı.

SOLID Prensipleri:
- SRP: build_model, compile_model, predict, get_embedding_model ayrı sorumluluklar.
- OCP: Yeni ağırlık / katman ekleme mevcut fonksiyonları bozmaz.
- DIP: CLASS_NAMES dışarıdan okunabilir; model yolu parametre olarak alınır.
"""

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ── Sabitler ─────────────────────────────────────────────────────────────

IMG_SIZE  = (224, 224)
IMG_SHAPE = (224, 224, 3)

CLASS_NAMES: list[str] = [
    "caretta_caretta",
    "chelonia_mydas",
    "eretmochelys_imbricata",
]

CLASS_LABELS_TR: dict[str, str] = {
    "caretta_caretta":        "Caretta Caretta",
    "chelonia_mydas":         "Yeşil Deniz Kaplumbağası",
    "eretmochelys_imbricata": "Hawksbill Kaplumbağası",
}

NUM_CLASSES = len(CLASS_NAMES)


# ── Model Oluşturma (SRP) ────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    ImageNet ağırlıklı EfficientNetB0 üzerine sınıflandırma başlığı ekler.
    Base başlangıçta dondurulmuştur; fine-tune için unfreeze_top() kullanılır.
    """
    base = EfficientNetB0(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="turtle_classifier")


def unfreeze_top(model: tf.keras.Model, n_layers: int = 40) -> tf.keras.Model:
    """
    Base modelin son n_layers katmanını fine-tune için açar (OCP).
    Mevcut model nesnesini değiştirir, yeni nesne döndürmez.
    """
    base = model.layers[1]          # Functional API'de 2. katman = EfficientNetB0
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Adam optimizer + categorical cross-entropy ile derler."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(
    save_path: str = "models/turtle_species_model.keras",
    patience_stop: int = 8,
    patience_lr: int = 3,
) -> list:
    """Eğitim callback listesi (SRP — konfigürasyon parametrik)."""
    return [
        callbacks.ModelCheckpoint(
            filepath=save_path,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-8,
            verbose=1,
        ),
    ]


# ── Veri Artırma (SRP) ───────────────────────────────────────────────────

def get_data_generators(
    dataset_dir: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    EfficientNet preprocess_input tabanlı train/val generator'ları döndürür.
    NOT: rescale=1/255 KULLANILMAZ; EfficientNet kendi normalizasyonunu içerir.
    """
    common_kwargs = dict(
        preprocessing_function=preprocess_input,
        validation_split=val_split,
    )

    train_gen = ImageDataGenerator(
        **common_kwargs,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.7, 1.3],
        zoom_range=0.3,
        horizontal_flip=True,
        shear_range=0.15,
        fill_mode="reflect",
    ).flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=seed,
    )

    val_gen = ImageDataGenerator(**common_kwargs).flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=seed,
    )

    return train_gen, val_gen


# ── Tahmin (SRP) ─────────────────────────────────────────────────────────

def load_model(model_path: str) -> tf.keras.Model | None:
    """Kaydedilmiş modeli yükler; başarısızda None döner."""
    try:
        m = tf.keras.models.load_model(model_path)
        print(f"[Model] Yüklendi: {model_path}")
        return m
    except Exception as exc:
        print(f"[Model] Yükleme hatası: {exc}")
        return None


def predict(model: tf.keras.Model, preprocessed_img: np.ndarray) -> dict:
    """
    (224,224,3) float32 görüntü → tahmin sözlüğü.

    Dönüş:
        {
            "class":      "caretta_caretta",
            "label_tr":   "Caretta Caretta",
            "confidence": 0.93,
            "all_scores": {"caretta_caretta": 0.93, ...},
        }
    """
    img_batch = np.expand_dims(preprocessed_img, axis=0)
    scores    = model.predict(img_batch, verbose=0)[0]

    best_idx  = int(np.argmax(scores))
    best_cls  = CLASS_NAMES[best_idx]

    return {
        "class":      best_cls,
        "label_tr":   CLASS_LABELS_TR.get(best_cls, best_cls),
        "confidence": float(scores[best_idx]),
        "all_scores": dict(zip(CLASS_NAMES, scores.tolist())),
    }


def get_embedding_model() -> tf.keras.Model:
    """
    1280-boyutlu yüz haritası çıkaran embedding modeli.
    Orijinal ImageNet ağırlıkları ile en yüksek genelleme başarımı sunar.
    """
    print("[Model] Orijinal ImageNet agirliklari yukleniyor (En yuksek basarim).")
    base = EfficientNetB0(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    return models.Sequential(
        [base, layers.GlobalAveragePooling2D()],
        name="embedding_extractor",
    )


def extract_embedding(
    embedding_model: tf.keras.Model,
    preprocessed_img: np.ndarray,
) -> np.ndarray:
    """Görsel → 1280-boyutlu numpy vektörü."""
    batch = np.expand_dims(preprocessed_img, axis=0)
    return embedding_model(batch, training=False).numpy()[0]
