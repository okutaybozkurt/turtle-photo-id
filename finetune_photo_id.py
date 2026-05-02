"""
finetune_photo_id.py
====================
EfficientNetB0'ı kaplumbağa bireyleri üzerinde fine-tune eder.
Embedding dostu mimari: BatchNorm/Dropout kaldırıldı, 
embedding katmanı doğrudan GAP çıkışından alınır.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks
from PIL import Image

# ── Sabitler ──────────────────────────────────────────────────────────────
KAGGLE_DIR     = "dataset_kaggle/images"
FINETUNED_PATH = "models/embedding_finetuned.keras"
IMG_SIZE       = (224, 224)
IMG_SHAPE      = (224, 224, 3)
BATCH_SIZE     = 32
MIN_PHOTOS     = 3
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Veri Hazırlama ────────────────────────────────────────────────────────

def load_dataset():
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    turtle_folders = sorted([
        d for d in os.listdir(KAGGLE_DIR)
        if os.path.isdir(os.path.join(KAGGLE_DIR, d))
    ])

    valid_folders = []
    for folder in turtle_folders:
        imgs = sorted(
            glob.glob(os.path.join(KAGGLE_DIR, folder, "*.JPG")) +
            glob.glob(os.path.join(KAGGLE_DIR, folder, "*.jpg")) +
            glob.glob(os.path.join(KAGGLE_DIR, folder, "*.jpeg")) +
            glob.glob(os.path.join(KAGGLE_DIR, folder, "*.png"))
        )
        if len(imgs) >= MIN_PHOTOS:
            valid_folders.append((folder, imgs))

    label_map = {folder: idx for idx, (folder, _) in enumerate(valid_folders)}
    num_classes = len(label_map)

    for folder, imgs in valid_folders:
        label = label_map[folder]
        val_paths.append(imgs[-1])
        val_labels.append(label)
        for img_path in imgs[:-1]:
            train_paths.append(img_path)
            train_labels.append(label)

    print(f"Egitilebilir birey sayisi : {num_classes}")
    print(f"Egitim seti               : {len(train_paths)} fotograf")
    print(f"Dogrulama seti            : {len(val_paths)} fotograf")
    return (train_paths, train_labels), (val_paths, val_labels), num_classes


def load_and_preprocess(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return preprocess_input(arr)


def make_tf_dataset(paths, labels, shuffle=True, augment=False):
    def generator():
        indices = list(range(len(paths)))
        if shuffle:
            random.shuffle(indices)
        for i in indices:
            img = load_and_preprocess(paths[i])
            yield img, labels[i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if augment:
        aug_layer = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ], name="augmentation")
        ds = ds.map(
            lambda x, y: (aug_layer(tf.expand_dims(x, 0), training=True)[0], y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model ─────────────────────────────────────────────────────────────────

def build_model(num_classes: int):
    """
    Embedding dostu mimari:
    EfficientNetB0 -> GAP (1280-d embedding) -> Dense (classifier)
    
    Fine-tuning sonrası GAP çıkışı direkt embedding olarak kullanılır.
    BatchNorm/Dropout yok — embedding uzayını bozmamak için.
    """
    base = EfficientNetB0(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base(inputs, training=False)
    embedding = layers.GlobalAveragePooling2D(name="embedding_output")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(embedding)

    model = tf.keras.Model(inputs, outputs, name="turtle_id_classifier")
    return model, base


def main():
    print("=" * 60)
    print("  Fine-Tuning: Kaplumbaga Birey Tanima (v2)")
    print("=" * 60)

    (train_p, train_l), (val_p, val_l), num_classes = load_dataset()
    train_ds = make_tf_dataset(train_p, train_l, shuffle=True, augment=True)
    val_ds   = make_tf_dataset(val_p, val_l, shuffle=False, augment=False)

    # ── Phase 1: Head Egitimi ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1: Base Donuk — Head Egitimi (15 epoch)")
    print("=" * 60)

    model, base = build_model(num_classes)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds, validation_data=val_ds, epochs=15,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=2, min_lr=1e-6, verbose=1),
        ],
        verbose=1,
    )
    p1_acc = model.evaluate(val_ds, verbose=0)[1]
    print(f"\n  Phase 1 — Val Accuracy: {p1_acc:.2%}")

    # ── Phase 2: Fine-Tuning ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: Fine-Tuning — Son 50 Katman Acik (20 epoch)")
    print("=" * 60)

    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"  Base: {len(base.layers)} katman, {trainable_count} egitilecek")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds, validation_data=val_ds, epochs=20,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                        patience=3, min_lr=1e-8, verbose=1),
        ],
        verbose=1,
    )
    p2_acc = model.evaluate(val_ds, verbose=0)[1]
    print(f"\n  Phase 2 — Val Accuracy: {p2_acc:.2%}")

    # ── Embedding Modeli Kaydet ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Embedding Modeli Cikartiliyor ve Kaydediliyor...")
    print("=" * 60)

    # Classifier head'i kes, sadece embedding katmanını al
    emb_layer = model.get_layer("embedding_output")
    emb_model = tf.keras.Model(
        inputs=model.input,
        outputs=emb_layer.output,
        name="embedding_finetuned",
    )

    os.makedirs("models", exist_ok=True)
    emb_model.save(FINETUNED_PATH)

    # Dogrulama
    test_img = load_and_preprocess(val_p[0])
    test_emb = emb_model(np.expand_dims(test_img, 0), training=False).numpy()[0]
    print(f"  Embedding boyutu: {test_emb.shape}")
    print(f"  Model kaydedildi: {FINETUNED_PATH}")

    print("\n" + "=" * 60)
    print("  TAMAMLANDI!")
    print(f"  Phase 1 Accuracy: {p1_acc:.2%}")
    print(f"  Phase 2 Accuracy: {p2_acc:.2%}")
    print()
    print("  Sonraki adimlar:")
    print("    1. python setup_photo_id.py    (DB embeddingleri guncelle)")
    print("    2. python evaluate_photo_id.py  (yeni basarim testi)")
    print("=" * 60)


if __name__ == "__main__":
    main()
