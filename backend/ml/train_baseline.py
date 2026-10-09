import os
import sys
import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE,
    PHASE1_EPOCHS, PHASE1_LR, PHASE1_PATIENCE,
    QUICK_EPOCHS, MODEL_KERAS_PATH, DATA_DIR
)
from augment import get_training_augmentations, get_validation_augmentations


def create_model(freeze_base: bool = True) -> tf.keras.Model:
    """Create MobileNetV2 model with custom head."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    if freeze_base:
        base_model.trainable = False
    
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model


def get_datasets(data_dir: Path, batch_size: int, quick: bool = False):
    """Create train/val datasets from directory structure."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Train/val directories not found at {data_dir}. "
            "Run download_data.py --organize first."
        )
    
    train_aug = get_training_augmentations()
    val_aug = get_validation_augmentations()
    
    def load_dataset(directory, augmentations, shuffle=True):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='int',
            class_names=CLASS_NAMES,
            color_mode='rgb',
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            shuffle=shuffle,
            seed=42
        )
        
        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if augmentations is not None:
            def augment(image, label):
                image_np = image.numpy()
                augmented = augmentations(image=image_np)
                return augmented['image'], label
            
            dataset = dataset.map(
                lambda x, y: tf.py_function(augment, [x, y], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_ds = load_dataset(train_dir, train_aug, shuffle=True)
    val_ds = load_dataset(val_dir, val_aug, shuffle=False)
    
    if quick:
        train_ds = train_ds.take(10)
        val_ds = val_ds.take(5)
    
    return train_ds, val_ds


def train_baseline(quick: bool = False, output_path: Path = None):
    """Phase 1: Train with frozen base model."""
    epochs = QUICK_EPOCHS if quick else PHASE1_EPOCHS
    
    print(f"=== Phase 1: Baseline Training (frozen base) ===")
    print(f"Epochs: {epochs}, LR: {PHASE1_LR}, Batch: {BATCH_SIZE}")
    
    train_ds, val_ds = get_datasets(DATA_DIR, BATCH_SIZE, quick)
    
    model = create_model(freeze_base=True)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=PHASE1_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    
    model.summary()
    
    checkpoint_path = output_path or MODEL_KERAS_PATH.with_suffix('.phase1.keras')
    
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=PHASE1_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(log_dir='./logs/phase1', histogram_freq=1)
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    model.save(checkpoint_path)
    print(f"Phase 1 model saved to {checkpoint_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline training")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with few epochs")
    parser.add_argument("--output", type=str, help="Output model path")
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model, history = train_baseline(quick=args.quick, output_path=output_path)
    
    print("\n=== Phase 1 Complete ===")
    print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()