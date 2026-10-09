import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODEL_KERAS_PATH, MODEL_PATH, IMAGE_SIZE, TFLITE_QUANTIZATION,
    TFLITE_TARGET_SIZE_MB, CLASS_NAMES
)
from augment import get_validation_augmentations


def load_keras_model(model_path: Path = None):
    """Load Keras model."""
    path = model_path or MODEL_KERAS_PATH
    model = models.load_model(path)
    return model


def get_representative_dataset(data_dir: Path, num_samples: int = 100):
    """Generate representative dataset for quantization calibration."""
    test_dir = data_dir / "test"
    
    if not test_dir.exists():
        print("Test directory not found, using random data for calibration")
        def gen():
            for _ in range(num_samples):
                yield [np.random.rand(1, *IMAGE_SIZE, 3).astype(np.float32)]
        return gen
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels=None,
        color_mode='rgb',
        batch_size=1,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=42
    )
    
    def preprocess(image):
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.take(num_samples)
    
    def gen():
        for image in dataset:
            yield [image.numpy()]
    
    return gen


def convert_to_tflite(
    model: tf.keras.Model,
    output_path: Path,
    quantization: str = TFLITE_QUANTIZATION,
    representative_data_gen=None
):
    """Convert Keras model to TFLite with quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        print("Converting to int8 quantized TFLite...")
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("Converting to float16 quantized TFLite...")
    elif quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Converting with dynamic range quantization...")
    else:
        print("Converting to float32 TFLite (no quantization)...")
    
    tflite_model = converter.convert()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved to {output_path} ({size_mb:.2f} MB)")
    
    return tflite_model, size_mb


def benchmark_tflite(model_path: Path, num_runs: int = 100, warmup: int = 10):
    """Benchmark TFLite model latency."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        input_data = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    else:
        input_data = np.random.rand(*input_shape).astype(np.float32)
    
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99))
    }


def benchmark_keras(model: tf.keras.Model, num_runs: int = 100, warmup: int = 10):
    """Benchmark Keras model latency."""
    input_data = np.random.rand(1, *IMAGE_SIZE, 3).astype(np.float32)
    
    for _ in range(warmup):
        model.predict(input_data, verbose=0)
    
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.predict(input_data, verbose=0)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99))
    }


def verify_tflite_output(model: tf.keras.Model, tflite_path: Path, num_samples: int = 10):
    """Verify TFLite output matches Keras output."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    max_diff = 0.0
    mean_diff = 0.0
    
    for _ in range(num_samples):
        input_data = np.random.rand(1, *IMAGE_SIZE, 3).astype(np.float32)
        
        keras_out = model.predict(input_data, verbose=0)
        
        if input_details[0]['dtype'] == np.uint8:
            input_quantized = (input_data * 255).astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], input_quantized)
        else:
            interpreter.set_tensor(input_details[0]['index'], input_data)
        
        interpreter.invoke()
        tflite_out = interpreter.get_tensor(output_details[0]['index'])
        
        if output_details[0]['dtype'] == np.uint8:
            tflite_out = tflite_out.astype(np.float32) / 255.0
        
        diff = np.abs(keras_out - tflite_out)
        max_diff = max(max_diff, np.max(diff))
        mean_diff += np.mean(diff)
    
    mean_diff /= num_samples
    
    print(f"Verification: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    return max_diff, mean_diff


def export_tflite(
    model_path: Path = None,
    output_path: Path = None,
    quantization: str = TFLITE_QUANTIZATION,
    data_dir: Path = None,
    benchmark: bool = True
):
    """Full export pipeline: convert, verify, benchmark."""
    model_path = model_path or MODEL_KERAS_PATH
    output_path = output_path or MODEL_PATH
    data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
    
    print(f"=== TFLite Export ===")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantization}")
    
    model = load_keras_model(model_path)
    
    rep_data_gen = get_representative_dataset(data_dir) if quantization == "int8" else None
    
    tflite_model, size_mb = convert_to_tflite(
        model, output_path, quantization, rep_data_gen
    )
    
    if size_mb > TFLITE_TARGET_SIZE_MB:
        print(f"WARNING: Model size ({size_mb:.2f} MB) exceeds target ({TFLITE_TARGET_SIZE_MB} MB)")
    else:
        print(f"Model size OK: {size_mb:.2f} MB <= {TFLITE_TARGET_SIZE_MB} MB")
    
    results = {
        "model_path": str(output_path),
        "quantization": quantization,
        "size_mb": size_mb,
        "target_size_mb": TFLITE_TARGET_SIZE_MB,
        "size_ok": size_mb <= TFLITE_TARGET_SIZE_MB
    }
    
    if quantization == "int8":
        max_diff, mean_diff = verify_tflite_output(model, output_path)
        results["verification"] = {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "passed": max_diff < 0.01
        }
    
    if benchmark:
        print("\nBenchmarking Keras model...")
        keras_bench = benchmark_keras(model)
        results["keras_latency"] = keras_bench
        print(f"  Mean: {keras_bench['mean_ms']:.2f} ms")
        
        print("Benchmarking TFLite model...")
        tflite_bench = benchmark_tflite(output_path)
        results["tflite_latency"] = tflite_bench
        print(f"  Mean: {tflite_bench['mean_ms']:.2f} ms")
        
        speedup = keras_bench['mean_ms'] / tflite_bench['mean_ms']
        results["speedup_factor"] = speedup
        print(f"  Speedup: {speedup:.2f}x")
    
    results_path = output_path.with_suffix('.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExport results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Export model to TFLite")
    parser.add_argument("--model", type=str, help="Path to Keras model")
    parser.add_argument("--output", type=str, help="Output TFLite path")
    parser.add_argument("--quantization", type=str, default=TFLITE_QUANTIZATION,
                        choices=["int8", "float16", "dynamic", "none"])
    parser.add_argument("--data-dir", type=str, help="Data directory for calibration")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmarking")
    args = parser.parse_args()
    
    export_tflite(
        model_path=Path(args.model) if args.model else None,
        output_path=Path(args.output) if args.output else None,
        quantization=args.quantization,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        benchmark=not args.no_benchmark
    )


if __name__ == "__main__":
    main()