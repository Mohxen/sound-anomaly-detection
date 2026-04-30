import argparse

from src.dataset import load_normal_train_and_mixed_test
from src.train_autoencoder import train_model
from src.tuning_autoencoder import (
    score_autoencoder_components,
    split_file_groups,
    tune_autoencoder_weights,
)
from src.evaluate_autoencoder import (
    compute_latent_center,
    compute_threshold,
    fit_component_scaler,
    summarize_file_components,
)
from src.evaluate_classifier import compute_binary_metrics
from src.config import (
    AUTOENCODER_THRESHOLD_PERCENTILES,
    AUTOENCODER_TUNE_WEIGHTS,
    AUTOENCODER_VALIDATION_RATIO,
    AUTOENCODER_WEIGHT_GRID,
    DATASET_PATH,
    LATENT_WEIGHT,
    RANDOM_SEED,
    RECON_WEIGHT,
    TEST_RATIO,
    THRESHOLD_PERCENTILE,
    TOP_K_FILE_CHUNKS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    args = parse_args()
    print("=== Method 1: Autoencoder anomaly detection ===")
    print(f"Dataset path: {args.dataset_path}")
    print("Loading dataset...")

    X_train, train_files, test_normal_files, test_anomaly_files = load_normal_train_and_mixed_test(
        args.dataset_path,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    print("Train shape:", X_train.shape)
    print("Train normal files:", len(train_files))
    print("Test normal files:", len(test_normal_files))
    print("Test anomaly files:", len(test_anomaly_files))

    if len(X_train) == 0:
        raise ValueError("No training chunks were created from normal audio.")
    if len(test_normal_files) == 0:
        raise ValueError("No normal test files were created.")
    if len(test_anomaly_files) == 0:
        raise ValueError("No abnormal/anomaly test files were created.")

    # Global normalization from TRAIN only
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
    print("Validation normal files:", len(validation_normal_files))
    print("Validation anomaly files:", len(validation_anomaly_files))
    print("Final test normal files:", len(final_normal_files))
    print("Final test anomaly files:", len(final_anomaly_files))

    # Train model
    print("Training model...")
    model = train_model(X_train)

    # Evaluate
    print("Evaluating...")
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
    final_normal_components = summarize_file_components(model, final_normal_files, latent_center, top_k=TOP_K_FILE_CHUNKS)
    final_anomaly_components = summarize_file_components(model, final_anomaly_files, latent_center, top_k=TOP_K_FILE_CHUNKS)

    component_scaler = fit_component_scaler(train_components)

    if AUTOENCODER_TUNE_WEIGHTS:
        best = tune_autoencoder_weights(
            train_components,
            validation_normal_components,
            validation_anomaly_components,
            component_scaler,
            weight_grid=AUTOENCODER_WEIGHT_GRID,
            threshold_percentiles=AUTOENCODER_THRESHOLD_PERCENTILES,
        )
        recon_weight = best["recon_weight"]
        latent_weight = best["latent_weight"]
        threshold_percentile = best["threshold_percentile"]
        print("Best validation tuning:")
        print(
            f"  recon_weight={recon_weight}, latent_weight={latent_weight}, "
            f"threshold_percentile={threshold_percentile}, threshold={best['threshold']:.6f}"
        )
        print(
            f"  validation FPR={best['metrics']['false_positive_rate']:.4f}, "
            f"Recall={best['metrics']['recall']:.4f}, F1={best['metrics']['f1']:.4f}"
        )
    else:
        recon_weight = RECON_WEIGHT
        latent_weight = LATENT_WEIGHT
        threshold_percentile = THRESHOLD_PERCENTILE

    train_scores = score_autoencoder_components(train_components, component_scaler, recon_weight, latent_weight)
    validation_normal_scores = score_autoencoder_components(
        validation_normal_components,
        component_scaler,
        recon_weight,
        latent_weight,
    )
    validation_anomaly_scores = score_autoencoder_components(
        validation_anomaly_components,
        component_scaler,
        recon_weight,
        latent_weight,
    )
    normal_scores = score_autoencoder_components(final_normal_components, component_scaler, recon_weight, latent_weight)
    anomaly_scores = score_autoencoder_components(final_anomaly_components, component_scaler, recon_weight, latent_weight)

    # Threshold is learned from normal training data only.
    print(
        f"Method 1 file-level score summary (top {TOP_K_FILE_CHUNKS} chunks per file, "
        f"recon_weight={recon_weight}, latent_weight={latent_weight}):"
    )
    print_component_summary("  Train components", train_components)
    print_component_summary("  Validation normal components", validation_normal_components)
    print_component_summary("  Validation anomaly components", validation_anomaly_components)
    print_component_summary("  Final test normal components", final_normal_components)
    print_component_summary("  Final test anomaly components", final_anomaly_components)
    print("  Train normal:", describe_errors(train_scores))
    print("  Validation normal :", describe_errors(validation_normal_scores))
    print("  Validation anomaly:", describe_errors(validation_anomaly_scores))
    print("  Final test normal :", describe_errors(normal_scores))
    print("  Final test anomaly:", describe_errors(anomaly_scores))

    print("Final test threshold sweep with selected weights:")
    for percentile in AUTOENCODER_THRESHOLD_PERCENTILES:
        candidate_threshold = compute_threshold(train_scores, percentile=percentile)
        candidate_fpr = sum(e > candidate_threshold for e in normal_scores) / len(normal_scores)
        candidate_tpr = sum(e > candidate_threshold for e in anomaly_scores) / len(anomaly_scores)
        print(
            f"  p{percentile}: threshold={candidate_threshold:.6f}, "
            f"FPR={candidate_fpr:.4f}, Recall={candidate_tpr:.4f}"
        )

    threshold = compute_threshold(train_scores, percentile=threshold_percentile)
    print("Threshold:", threshold)

    # Predictions
    pred_normal = [e > threshold for e in normal_scores]
    pred_anomaly = [e > threshold for e in anomaly_scores]

    # Metrics
    false_positive_rate = sum(pred_normal) / len(pred_normal)
    true_positive_rate = sum(pred_anomaly) / len(pred_anomaly)
    metrics = compute_binary_metrics(normal_scores, anomaly_scores, threshold=threshold)

    print("False Positive Rate:", false_positive_rate)
    print("True Positive Rate (Recall):", true_positive_rate)
    print("Precision:", metrics["precision"])
    print("F1:", metrics["f1"])

    # Plot
    plt.hist(normal_scores, bins=50, alpha=0.5, label="Normal")
    plt.hist(anomaly_scores, bins=50, alpha=0.5, label="Anomaly")
    plt.legend()
    plt.title("Method 1 autoencoder file-level anomaly score")
    plt.savefig("method1_autoencoder_file_level_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plot: method1_autoencoder_file_level_scores.png")
    print("Done!")

def describe_errors(errors):
    errors = np.array(errors)
    return (
        f"mean={errors.mean():.6f}, std={errors.std():.6f}, "
        f"p50={np.percentile(errors, 50):.6f}, "
        f"p75={np.percentile(errors, 75):.6f}, "
        f"p95={np.percentile(errors, 95):.6f}"
    )


def normalize_file_groups(file_groups, mean, std):
    return [(features - mean) / std for features in file_groups]


def print_component_summary(label, components):
    recon = [item["recon"] for item in components]
    latent = [item["latent"] for item in components]
    print(f"{label}: recon({describe_errors(recon)}), latent({describe_errors(latent)})")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Method 1: autoencoder anomaly detection.")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Path to one dataset ID folder.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
