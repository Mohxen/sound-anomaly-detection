import argparse
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import (
    AUTOENCODER_THRESHOLD_PERCENTILES,
    AUTOENCODER_TUNE_WEIGHTS,
    AUTOENCODER_VALIDATION_RATIO,
    AUTOENCODER_WEIGHT_GRID,
    CLASSIFIER_FILE_TOP_K,
    CLASSIFIER_THRESHOLD,
    DATASET_PATH,
    LATENT_WEIGHT,
    RANDOM_SEED,
    RECON_WEIGHT,
    TEST_RATIO,
    THRESHOLD_PERCENTILE,
    TOP_K_FILE_CHUNKS,
)
from src.tuning_autoencoder import (
    score_autoencoder_components,
    split_file_groups,
    tune_autoencoder_weights,
)
from src.dataset import load_normal_train_and_mixed_test, load_supervised_file_splits
from src.evaluate_autoencoder import (
    compute_latent_center,
    compute_threshold,
    fit_component_scaler,
    summarize_file_components,
)
from src.evaluate_classifier import compute_binary_metrics, compute_file_probabilities
from src.features import extract_logmel
from src.model import CNNBinaryClassifier, CNNAutoencoder
from src.train_autoencoder import train_model
from src.train_classifier import train_classifier


def main():
    args = parse_args()
    print(f"Dataset path: {args.dataset_path}")
    print("=== Method 1: Autoencoder anomaly detection ===")
    autoencoder_result = run_autoencoder_method(args.dataset_path)

    print("\n=== Method 2: Supervised CNN classifier ===")
    classifier_result = run_classifier_method(args.dataset_path)

    print("\n=== Inference benchmark ===")
    benchmark = benchmark_inference(autoencoder_result["model"], classifier_result["model"])

    print("\n=== Summary ===")
    print_summary("Autoencoder", autoencoder_result, benchmark["autoencoder"])
    print_summary("Supervised CNN", classifier_result, benchmark["classifier"])
    print("Extra CNN inference time per 10-second file:", format_seconds(
        benchmark["classifier"]["model_seconds"] - benchmark["autoencoder"]["model_seconds"]
    ))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plot_path = results_dir / "methods_autoencoder_vs_supervised_cnn_comparison.png"
    plot_comparison(autoencoder_result, classifier_result, plot_path)
    print(f"Saved comparison plot: {plot_path}")


def run_autoencoder_method(dataset_path):
    X_train, train_files, test_normal_files, test_anomaly_files = load_normal_train_and_mixed_test(
        dataset_path,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    mean = X_train.mean()
    std = X_train.std() + 1e-8
    X_train = (X_train - mean) / std
    train_files = normalize_file_groups(train_files, mean, std)
    test_normal_files = normalize_file_groups(test_normal_files, mean, std)
    test_anomaly_files = normalize_file_groups(test_anomaly_files, mean, std)
    validation_normal_files, final_normal_files = split_file_groups(
        test_normal_files,
        validation_ratio=AUTOENCODER_VALIDATION_RATIO,
        seed=RANDOM_SEED,
    )
    validation_anomaly_files, final_anomaly_files = split_file_groups(
        test_anomaly_files,
        validation_ratio=AUTOENCODER_VALIDATION_RATIO,
        seed=RANDOM_SEED,
    )

    start = time.perf_counter()
    model = train_model(X_train)
    train_seconds = time.perf_counter() - start

    latent_center = compute_latent_center(model, X_train)
    train_components = summarize_file_components(model, train_files, latent_center, top_k=TOP_K_FILE_CHUNKS)
    validation_normal_components = summarize_file_components(
        model,
        validation_normal_files,
        latent_center,
        top_k=TOP_K_FILE_CHUNKS,
    )
    validation_anomaly_components = summarize_file_components(
        model,
        validation_anomaly_files,
        latent_center,
        top_k=TOP_K_FILE_CHUNKS,
    )
    normal_components = summarize_file_components(model, final_normal_files, latent_center, top_k=TOP_K_FILE_CHUNKS)
    anomaly_components = summarize_file_components(model, final_anomaly_files, latent_center, top_k=TOP_K_FILE_CHUNKS)

    scaler = fit_component_scaler(train_components)

    if AUTOENCODER_TUNE_WEIGHTS:
        best = tune_autoencoder_weights(
            train_components,
            validation_normal_components,
            validation_anomaly_components,
            scaler,
            weight_grid=AUTOENCODER_WEIGHT_GRID,
            threshold_percentiles=AUTOENCODER_THRESHOLD_PERCENTILES,
        )
        recon_weight = best["recon_weight"]
        latent_weight = best["latent_weight"]
        threshold_percentile = best["threshold_percentile"]
        print(
            "Method 1 tuned weights: "
            f"recon={recon_weight}, latent={latent_weight}, "
            f"threshold_percentile={threshold_percentile}"
        )
        print(
            "Method 1 validation metrics: "
            f"FPR={best['metrics']['false_positive_rate']:.4f}, "
            f"Recall={best['metrics']['recall']:.4f}, "
            f"F1={best['metrics']['f1']:.4f}"
        )
    else:
        recon_weight = RECON_WEIGHT
        latent_weight = LATENT_WEIGHT
        threshold_percentile = THRESHOLD_PERCENTILE

    train_scores = score_autoencoder_components(train_components, scaler, recon_weight, latent_weight)
    normal_scores = score_autoencoder_components(normal_components, scaler, recon_weight, latent_weight)
    anomaly_scores = score_autoencoder_components(anomaly_components, scaler, recon_weight, latent_weight)

    threshold = compute_threshold(train_scores, percentile=threshold_percentile)
    metrics = compute_binary_metrics(normal_scores, anomaly_scores, threshold=threshold)

    print("Method 1 validation normal files:", len(validation_normal_files))
    print("Method 1 validation anomaly files:", len(validation_anomaly_files))
    print("Method 1 final test normal files:", len(final_normal_files))
    print("Method 1 final test anomaly files:", len(final_anomaly_files))
    print("Method 1 threshold percentile:", threshold_percentile)
    print("Method 1 threshold:", threshold)
    print_metrics(metrics)
    print("Training time:", format_seconds(train_seconds))

    return {
        "model": model,
        "normal_scores": normal_scores,
        "anomaly_scores": anomaly_scores,
        "threshold": threshold,
        "recon_weight": recon_weight,
        "latent_weight": latent_weight,
        "threshold_percentile": threshold_percentile,
        "metrics": metrics,
        "train_seconds": train_seconds,
        "params": count_parameters(model),
    }


def run_classifier_method(dataset_path):
    (
        X_train,
        y_train,
        _train_normal_files,
        _train_anomaly_files,
        test_normal_files,
        test_anomaly_files,
    ) = load_supervised_file_splits(dataset_path, test_ratio=TEST_RATIO, seed=RANDOM_SEED)

    mean = X_train.mean()
    std = X_train.std() + 1e-8
    X_train = (X_train - mean) / std
    test_normal_files = normalize_file_groups(test_normal_files, mean, std)
    test_anomaly_files = normalize_file_groups(test_anomaly_files, mean, std)

    start = time.perf_counter()
    model = train_classifier(X_train, y_train)
    train_seconds = time.perf_counter() - start

    normal_scores = compute_file_probabilities(model, test_normal_files, top_k=CLASSIFIER_FILE_TOP_K)
    anomaly_scores = compute_file_probabilities(model, test_anomaly_files, top_k=CLASSIFIER_FILE_TOP_K)
    metrics = compute_binary_metrics(normal_scores, anomaly_scores, threshold=CLASSIFIER_THRESHOLD)

    print("Method 2 test normal files:", len(test_normal_files))
    print("Method 2 test anomaly files:", len(test_anomaly_files))
    print("Method 2 threshold:", CLASSIFIER_THRESHOLD)
    print_metrics(metrics)
    print("Training time:", format_seconds(train_seconds))

    return {
        "model": model,
        "normal_scores": normal_scores,
        "anomaly_scores": anomaly_scores,
        "threshold": CLASSIFIER_THRESHOLD,
        "metrics": metrics,
        "train_seconds": train_seconds,
        "params": count_parameters(model),
    }


def benchmark_inference(autoencoder, classifier):
    sample = np.zeros((10, 16000), dtype=np.float32)

    start = time.perf_counter()
    features = extract_logmel(sample)
    feature_seconds = time.perf_counter() - start

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    autoencoder.eval()
    classifier.eval()

    with torch.no_grad():
        autoencoder(x)
        classifier(x)

    loops = 200
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(loops):
            autoencoder(x)
        autoencoder_seconds = (time.perf_counter() - start) / loops

        start = time.perf_counter()
        for _ in range(loops):
            classifier(x)
        classifier_seconds = (time.perf_counter() - start) / loops

    return {
        "feature_seconds": feature_seconds,
        "autoencoder": {
            "model_seconds": autoencoder_seconds,
            "total_seconds": feature_seconds + autoencoder_seconds,
        },
        "classifier": {
            "model_seconds": classifier_seconds,
            "total_seconds": feature_seconds + classifier_seconds,
        },
    }


def print_summary(name, result, benchmark):
    metrics = result["metrics"]
    print(f"{name}:")
    print(f"  Parameters: {result['params']}")
    print(f"  Training time: {format_seconds(result['train_seconds'])}")
    print(f"  Inference model time per 10-second file: {format_seconds(benchmark['model_seconds'])}")
    print(f"  Inference total time per 10-second file: {format_seconds(benchmark['total_seconds'])}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  FPR: {metrics['false_positive_rate']:.4f}")


def print_metrics(metrics):
    print(f"Confusion matrix: TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("False Positive Rate:", metrics["false_positive_rate"])


def plot_comparison(autoencoder_result, classifier_result, output_path):
    names = ["Autoencoder", "Supervised CNN"]
    recalls = [autoencoder_result["metrics"]["recall"], classifier_result["metrics"]["recall"]]
    fprs = [
        autoencoder_result["metrics"]["false_positive_rate"],
        classifier_result["metrics"]["false_positive_rate"],
    ]
    f1_scores = [autoencoder_result["metrics"]["f1"], classifier_result["metrics"]["f1"]]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, recalls, width, label="Recall")
    plt.bar(x, fprs, width, label="FPR")
    plt.bar(x + width, f1_scores, width, label="F1")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Method comparison: autoencoder vs supervised CNN")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def normalize_file_groups(file_groups, mean, std):
    return [(features - mean) / std for features in file_groups]


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def format_seconds(seconds):
    return f"{seconds:.6f}s"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare autoencoder and supervised CNN methods.")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Path to one dataset ID folder.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
