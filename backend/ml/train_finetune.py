import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE,
    PHASE2_EPOCHS, PHASE2_LR, PHASE2_PATIENCE, UNFREEZE_LAYERS,
    QUICK_EPOCHS, MODEL_KERAS_PATH, DATA_DIR
)
from augment import get_training_augmentations, get_validation_augmentations


def create_model_from_checkpoint(checkpoint_path: Path, unfreeze_layers: int = 30) -> tf.keras.Model:
    """Load model from checkpoint and unfreeze last N layers."""
    model = tf.keras.models.load_model(checkpoint_path)
    
    base_model = model.layers[1]
    
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    
    print(f"Unfroze last {unfreeze_layers} layers of base model")
    print(f"Total trainable params: {model.count_params():,}")
    
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


def train_finetune(
    checkpoint_path: Path,
    quick: bool = False,
    output_path: Path = None,
    unfreeze_layers: int = UNFREEZE_LAYERS
):
    """Phase 2: Fine-tune with unfrozen layers."""
    epochs = QUICK_EPOCHS if quick else PHASE2_EPOCHS
    
    print(f"=== Phase 2: Fine-tuning (unfreeze last {unfreeze_layers} layers) ===")
    print(f"Epochs: {epochs}, LR: {PHASE2_LR}, Batch: {BATCH_SIZE}")
    
    train_ds, val_ds = get_datasets(DATA_DIR, BATCH_SIZE, quick)
    
    model = create_model_from_checkpoint(checkpoint_path, unfreeze_layers)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=PHASE2_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    
    model.summary()
    
    checkpoint_path_out = output_path or MODEL_KERAS_PATH
    
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=PHASE2_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path_out,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),
        callbacks.TensorBoard(log_dir='./logs/phase2', histogram_freq=1)
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    model.save(checkpoint_path_out)
    print(f"Phase 2 model saved to {checkpoint_path_out}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tuning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Phase 1 checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with few epochs")
    parser.add_argument("--output", type=str, help="Output model path")
    parser.add_argument("--unfreeze", type=int, default=UNFREEZE_LAYERS, help="Number of layers to unfreeze")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    output_path = Path(args.output) if args.output else None
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model, history = train_finetune(
        checkpoint_path=checkpoint_path,
        quick=args.quick,
        output_path=output_path,
        unfreeze_layers=args.unfreeze
    )
    
    print("\n=== Phase 2 Complete ===")
    print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()