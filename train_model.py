"""
train_model.py
==============
EfficientNetB0 ile iki aşamalı eğitim.
KRİTİK: EfficientNet kendi normalizasyonunu içerir, rescale=1/255 KULLANILMAZ.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.utils import count_images_per_class, plot_training_history, ensure_dir

# ── Sabitler ───────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
MODEL_SAVE   = "models/turtle_species_model.keras"
HISTORY_PLOT = "models/training_history.png"
IMG_SIZE     = (224, 224)
IMG_SHAPE    = (224, 224, 3)
BATCH_SIZE   = 16
VAL_SPLIT    = 0.2
SEED         = 42


def make_generators():
    """
    EfficientNet için DOĞRU generator:
    - rescale YOK (EfficientNet kendi normalizasyonunu yapar)
    - preprocessing_function = preprocess_input
    """
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,   # EfficientNet'e özel
        validation_split=VAL_SPLIT,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.7, 1.3],
        zoom_range=0.3,
        horizontal_flip=True,
        shear_range=0.15,
        fill_mode="reflect",
    ).flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,   # Validation da aynı ön işleme
        validation_split=VAL_SPLIT,
    ).flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    return train_gen, val_gen


def build_model(num_classes: int, fine_tune_at: int = 0):
    """
    EfficientNetB0 + Functional API.
    fine_tune_at: Bu indeksten sonraki base katmanlar açılır (0 = hepsi dondurulmuş).
    """
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=IMG_SHAPE)
    base.trainable = False  # Başlangıçta tamamen dondurulmuş

    inputs  = tf.keras.Input(shape=IMG_SHAPE)
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.5)(x)
    x       = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs=x, name="turtle_classifier"), base


def main():
    print("=" * 65)
    print("  🐢 EfficientNetB0 — Düzeltilmiş İki Aşamalı Eğitim")
    print("=" * 65)

    counts = count_images_per_class(DATASET_DIR)
    if not counts:
        print(f"❌ Dataset boş: {DATASET_DIR}")
        sys.exit(1)

    num_classes = len(counts)
    total = sum(counts.values())
    print(f"\n📂 Veri seti: {total} görsel, {num_classes} sınıf")
    for cls, cnt in counts.items():
        print(f"   ✅ {cls}: {cnt}")

    ensure_dir("models")
    train_gen, val_gen = make_generators()
    print(f"\nSınıf sırası: {train_gen.class_indices}\n")

    # ── AŞAMA 1: Frozen Base (20 epoch) ────────────────────────────────────
    print("🚀 AŞAMA 1 — Base Dondurulmuş, Üst Katmanlar Eğitiliyor")
    model, base = build_model(num_classes)

    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=20,
        callbacks=[
            callbacks.ModelCheckpoint(MODEL_SAVE, save_best_only=True,
                                       monitor="val_accuracy", verbose=1),
            callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                     restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=3, min_lr=1e-6, verbose=1),
        ],
        verbose=1,
    )

    best1 = max(h1.history["val_accuracy"])
    print(f"\n✅ Aşama 1 bitti — En iyi val_accuracy: {best1:.2%}")

    # ── AŞAMA 2: Fine-Tuning (30 epoch) ────────────────────────────────────
    print("\n🚀 AŞAMA 2 — Fine-Tuning (base son 40 katman açıldı)")

    base.trainable = True
    # İlk N katmanı dondur, son 40'ı aç
    for layer in base.layers[:-40]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"   Base model: {len(base.layers)} katman, {trainable_count} eğitilebilir")

    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=30,
        callbacks=[
            callbacks.ModelCheckpoint(MODEL_SAVE, save_best_only=True,
                                       monitor="val_accuracy", verbose=1),
            callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                     restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                         patience=4, min_lr=1e-8, verbose=1),
        ],
        verbose=1,
    )

    best2 = max(h2.history["val_accuracy"])
    print(f"\n✅ Aşama 2 bitti — En iyi val_accuracy: {best2:.2%}")

    # Grafik
    plot_training_history(h2, save_path=HISTORY_PLOT)

    print(f"\n📊 Model  : {MODEL_SAVE}")
    print(f"📊 Grafik : {HISTORY_PLOT}")
    print(f"\n🎯 Final val_accuracy: {max(best1, best2):.2%}")
    print("Sonraki adım: streamlit run app.py")


if __name__ == "__main__":
    main()
