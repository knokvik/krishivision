import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE,
    MODEL_KERAS_PATH, DATA_DIR, PLANT_DOC_PATH
)
from augment import get_validation_augmentations


def load_model(model_path: Path = None):
    """Load trained model."""
    path = model_path or MODEL_KERAS_PATH
    model = models.load_model(path)
    return model


def get_test_dataset(data_dir: Path, split: str = "test", batch_size: int = BATCH_SIZE):
    """Load test dataset."""
    test_dir = data_dir / split
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    val_aug = get_validation_augmentations()
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
        seed=42
    )
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def get_plantdoc_dataset(batch_size: int = BATCH_SIZE):
    """Load PlantDoc dataset for field evaluation."""
    plantdoc_dir = PLANT_DOC_PATH / "PlantDoc-Object-Detection-Dataset-main" / "train"
    
    if not plantdoc_dir.exists():
        plantdoc_dir = PLANT_DOC_PATH / "train"
    
    if not plantdoc_dir.exists():
        print("PlantDoc dataset not found. Skipping field evaluation.")
        return None
    
    val_aug = get_validation_augmentations()
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        plantdoc_dir,
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
        seed=42
    )
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def evaluate_model(model, dataset, dataset_name: str = "Test"):
    """Evaluate model on a dataset."""
    print(f"\n=== Evaluating on {dataset_name} ===")
    
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_pred_proba.extend(predictions)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }


def plot_confusion_matrix(cm, class_names, output_path: Path, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm, annot=False, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_per_class_metrics(report, output_path: Path):
    """Plot per-class precision, recall, F1."""
    classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1')
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Per-class metrics plot saved to {output_path}")


def run_ablation_study(data_dir: Path, output_dir: Path):
    """Run ablation study: baseline / +bg-augment / +finetune."""
    print("\n=== Ablation Study ===")
    print("Note: This requires multiple trained models.")
    print("Expected models:")
    print("  1. Baseline (frozen base)")
    print("  2. Baseline + Background Augmentation")
    print("  3. Fine-tuned (unfreeze last 30 layers)")
    
    results = {}
    
    model_paths = {
        "baseline": output_dir / "model.phase1.keras",
        "baseline_bg_aug": output_dir / "model.phase1.bg_aug.keras",
        "finetuned": output_dir / "model.keras"
    }
    
    for name, path in model_paths.items():
        if path.exists():
            print(f"\nEvaluating {name}...")
            model = load_model(path)
            test_ds = get_test_dataset(data_dir)
            results[name] = evaluate_model(model, test_ds, name)
        else:
            print(f"Model not found: {path}")
    
    if results:
        ablation_path = output_dir / "ablation_results.json"
        with open(ablation_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAblation results saved to {ablation_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Data directory")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--plantdoc", action="store_true", help="Evaluate on PlantDoc")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    model = load_model(Path(args.model) if args.model else None)
    
    test_ds = get_test_dataset(data_dir)
    results = evaluate_model(model, test_ds, "PlantVillage Test")
    
    if args.plantdoc:
        plantdoc_ds = get_plantdoc_dataset()
        if plantdoc_ds:
            plantdoc_results = evaluate_model(model, plantdoc_ds, "PlantDoc Field")
            results["plantdoc"] = plantdoc_results
    
    if args.ablation:
        ablation_results = run_ablation_study(data_dir, output_dir)
        results["ablation"] = ablation_results
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_confusion_matrix(
        np.array(results["confusion_matrix"]),
        CLASS_NAMES,
        output_dir / "confusion_matrix.png",
        "PlantVillage Test - Confusion Matrix"
    )
    
    plot_per_class_metrics(
        results["classification_report"],
        output_dir / "per_class_metrics.png"
    )
    
    print(f"\n=== Summary ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()