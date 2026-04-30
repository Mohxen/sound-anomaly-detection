import argparse
import csv
from pathlib import Path

from compare_methods import (
    benchmark_inference,
    format_seconds,
    plot_comparison,
    run_autoencoder_method,
    run_classifier_method,
)
from src.config import DATASET_PATH


def main():
    args = parse_args()
    fan_root = Path(args.dataset_root)
    id_dirs = sorted(path for path in fan_root.glob("id_*") if path.is_dir())

    if not id_dirs:
        raise FileNotFoundError(f"No id_* folders found under: {fan_root}")

    rows = []
    for dataset_path in id_dirs:
        print(f"\n================ {dataset_path.name} ================")
        print("=== Method 1: Autoencoder anomaly detection ===")
        autoencoder_result = run_autoencoder_method(str(dataset_path))

        print("\n=== Method 2: Supervised CNN classifier ===")
        classifier_result = run_classifier_method(str(dataset_path))

        print("\n=== Inference benchmark ===")
        benchmark = benchmark_inference(autoencoder_result["model"], classifier_result["model"])

        plot_name = f"methods_autoencoder_vs_supervised_cnn_comparison_{dataset_path.name}.png"
        plot_comparison(autoencoder_result, classifier_result, plot_name)
        print(f"Saved comparison plot: {plot_name}")

        rows.extend(
            [
                build_row(dataset_path.name, "autoencoder", autoencoder_result, benchmark["autoencoder"]),
                build_row(dataset_path.name, "supervised_cnn", classifier_result, benchmark["classifier"]),
            ]
        )

    output_path = "all_ids_method_comparison.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV summary: {output_path}")
    print_summary_table(rows)


def build_row(dataset_id, method, result, benchmark):
    metrics = result["metrics"]
    return {
        "dataset_id": dataset_id,
        "method": method,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_positive_rate": metrics["false_positive_rate"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "threshold": result["threshold"],
        "recon_weight": result.get("recon_weight", ""),
        "latent_weight": result.get("latent_weight", ""),
        "threshold_percentile": result.get("threshold_percentile", ""),
        "parameters": result["params"],
        "training_seconds": result["train_seconds"],
        "inference_model_seconds_per_10s_file": benchmark["model_seconds"],
        "inference_total_seconds_per_10s_file": benchmark["total_seconds"],
    }


def print_summary_table(rows):
    print("\nSummary:")
    for row in rows:
        print(
            f"{row['dataset_id']} {row['method']}: "
            f"recall={row['recall']:.4f}, "
            f"FPR={row['false_positive_rate']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"inference={format_seconds(row['inference_total_seconds_per_10s_file'])}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare both methods across all id_* dataset folders.")
    parser.add_argument(
        "--dataset-root",
        default=str(Path(DATASET_PATH).parent),
        help="Path to folder containing id_* dataset folders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
