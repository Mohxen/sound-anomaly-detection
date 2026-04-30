import numpy as np

from src.evaluate_autoencoder import combine_file_scores, compute_threshold
from src.evaluate_classifier import compute_binary_metrics


def split_file_groups(file_groups, validation_ratio=0.5, seed=42):
    if len(file_groups) < 2:
        return file_groups, file_groups

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(file_groups))
    shuffled = [file_groups[index] for index in indices]
    validation_count = max(1, int(round(len(shuffled) * validation_ratio)))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[:validation_count], shuffled[validation_count:]


def score_autoencoder_components(components, scaler, recon_weight, latent_weight):
    return combine_file_scores(
        components,
        scaler,
        recon_weight=recon_weight,
        latent_weight=latent_weight,
    )


def tune_autoencoder_weights(
    train_components,
    validation_normal_components,
    validation_anomaly_components,
    scaler,
    weight_grid,
    threshold_percentiles,
):
    best = None

    for recon_weight, latent_weight in weight_grid:
        train_scores = score_autoencoder_components(train_components, scaler, recon_weight, latent_weight)
        normal_scores = score_autoencoder_components(validation_normal_components, scaler, recon_weight, latent_weight)
        anomaly_scores = score_autoencoder_components(validation_anomaly_components, scaler, recon_weight, latent_weight)

        for percentile in threshold_percentiles:
            threshold = compute_threshold(train_scores, percentile=percentile)
            metrics = compute_binary_metrics(normal_scores, anomaly_scores, threshold=threshold)
            candidate = {
                "recon_weight": recon_weight,
                "latent_weight": latent_weight,
                "threshold_percentile": percentile,
                "threshold": threshold,
                "metrics": metrics,
            }

            if best is None or is_better(candidate["metrics"], best["metrics"]):
                best = candidate

    return best


def is_better(candidate, current):
    candidate_key = (
        candidate["f1"],
        candidate["recall"],
        -candidate["false_positive_rate"],
        candidate["precision"],
    )
    current_key = (
        current["f1"],
        current["recall"],
        -current["false_positive_rate"],
        current["precision"],
    )
    return candidate_key > current_key
