import argparse

from src.config import (
    CLASSIFIER_FILE_TOP_K,
    CLASSIFIER_THRESHOLD,
    DATASET_PATH,
    RANDOM_SEED,
    TEST_RATIO,
)
from src.dataset import load_supervised_file_splits
from src.evaluate_classifier import compute_binary_metrics, compute_file_probabilities
from src.train_classifier import train_classifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    args = parse_args()
    print("=== Method 2: Supervised CNN classifier ===")
    print(f"Dataset path: {args.dataset_path}")
    print("Loading supervised dataset...")
    (
        X_train,
        y_train,
        train_normal_files,
        train_anomaly_files,
        test_normal_files,
        test_anomaly_files,
    ) = load_supervised_file_splits(args.dataset_path, test_ratio=TEST_RATIO, seed=RANDOM_SEED)

    print("Train chunks:", X_train.shape)
    print("Train labels:", np.bincount(y_train.astype(int)))
    print("Train normal files:", len(train_normal_files))
    print("Train anomaly files:", len(train_anomaly_files))
    print("Test normal files:", len(test_normal_files))
    print("Test anomaly files:", len(test_anomaly_files))

    mean = X_train.mean()
    std = X_train.std() + 1e-8
    X_train = (X_train - mean) / std
    test_normal_files = normalize_file_groups(test_normal_files, mean, std)
    test_anomaly_files = normalize_file_groups(test_anomaly_files, mean, std)

    print("Training classifier...")
    model = train_classifier(X_train, y_train)

    print("Evaluating classifier...")
    normal_scores = compute_file_probabilities(model, test_normal_files, top_k=CLASSIFIER_FILE_TOP_K)
    anomaly_scores = compute_file_probabilities(model, test_anomaly_files, top_k=CLASSIFIER_FILE_TOP_K)

    print(f"Method 2 file-level probability summary (top {CLASSIFIER_FILE_TOP_K} chunks per file):")
    print("  Test normal :", describe_scores(normal_scores))
    print("  Test anomaly:", describe_scores(anomaly_scores))

    print("Threshold sweep:")
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        metrics = compute_binary_metrics(normal_scores, anomaly_scores, threshold=threshold)
        print(
            f"  t={threshold:.1f}: "
            f"FPR={metrics['false_positive_rate']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    metrics = compute_binary_metrics(normal_scores, anomaly_scores, threshold=CLASSIFIER_THRESHOLD)
    print("Selected threshold:", CLASSIFIER_THRESHOLD)
    print("Confusion matrix:")
    print(f"  TN={metrics['tn']} FP={metrics['fp']}")
    print(f"  FN={metrics['fn']} TP={metrics['tp']}")
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("False Positive Rate:", metrics["false_positive_rate"])

    plt.hist(normal_scores, bins=40, alpha=0.5, label="Normal")
    plt.hist(anomaly_scores, bins=40, alpha=0.5, label="Abnormal")
    plt.axvline(CLASSIFIER_THRESHOLD, color="black", linestyle="--", label="Threshold")
    plt.legend()
    plt.title("Method 2 supervised CNN file probabilities")
    plt.savefig("method2_supervised_cnn_file_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plot: method2_supervised_cnn_file_probabilities.png")
    print("Done!")


def normalize_file_groups(file_groups, mean, std):
    return [(features - mean) / std for features in file_groups]


def describe_scores(scores):
    scores = np.array(scores)
    return (
        f"mean={scores.mean():.6f}, std={scores.std():.6f}, "
        f"p50={np.percentile(scores, 50):.6f}, "
        f"p75={np.percentile(scores, 75):.6f}, "
        f"p95={np.percentile(scores, 95):.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Method 2: supervised CNN classifier.")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Path to one dataset ID folder.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
