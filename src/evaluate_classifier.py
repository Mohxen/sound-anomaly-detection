import numpy as np
import torch


def compute_chunk_probabilities(model, X):
    probabilities = []

    model.eval()
    with torch.no_grad():
        for x in X:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            probability = torch.sigmoid(model(x)).item()
            probabilities.append(probability)

    return probabilities


def compute_file_probabilities(model, file_features, top_k=3):
    scores = []

    for features in file_features:
        probabilities = compute_chunk_probabilities(model, features)
        k = min(top_k, len(probabilities))
        scores.append(float(np.mean(sorted(probabilities, reverse=True)[:k])))

    return scores


def compute_binary_metrics(normal_scores, anomaly_scores, threshold=0.5):
    pred_normal = [score >= threshold for score in normal_scores]
    pred_anomaly = [score >= threshold for score in anomaly_scores]

    tn = sum(not pred for pred in pred_normal)
    fp = sum(pred_normal)
    fn = sum(not pred for pred in pred_anomaly)
    tp = sum(pred_anomaly)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    false_positive_rate = fp / (fp + tn) if fp + tn else 0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }
