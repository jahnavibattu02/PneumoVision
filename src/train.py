import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data import make_generators
from src.utils import ensure_dir, save_json

def build_model(img_size=(224, 224), lr=1e-3):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size[0], img_size[1], 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model, base

def get_class_weights(train_gen):
    y = train_gen.classes
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root containing train/val/test")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_frozen", type=int, default=5)
    parser.add_argument("--epochs_ft", type=int, default=5)
    parser.add_argument("--out_model", type=str, default="artifacts/model.keras")
    args = parser.parse_args()

    ensure_dir("artifacts")
    ensure_dir("reports")

    img_size = (args.img_size, args.img_size)
    train_gen, val_gen, test_gen = make_generators(args.data_dir, img_size=img_size, batch_size=args.batch_size)

    # Save labels
    # Keras assigns class indices alphabetically from folder names
    labels = {v: k for k, v in train_gen.class_indices.items()}
    save_json({"class_indices": train_gen.class_indices, "labels": labels}, "artifacts/labels.json")

    class_weights = get_class_weights(train_gen)

    model, base = build_model(img_size=img_size, lr=1e-3)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_roc_auc", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.out_model, monitor="val_roc_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("Phase A: training with frozen backbone...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs_frozen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("Phase B: fine-tuning top layers...")
    base.trainable = True
    # Fine-tune only last ~30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs_ft,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Saved best model to: {args.out_model}")
    print("Done.")

if __name__ == "__main__":
    main()
