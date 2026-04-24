from src.dataset import load_train_data, load_test_data
from src.train import train_model
from src.evaluate import compute_error, compute_threshold
from src.config import DATASET_PATH

import matplotlib.pyplot as plt

def main():
    print("Loading dataset...")

    train_path = DATASET_PATH + "/train"
    test_path  = DATASET_PATH + "/test"

    # Load data
    X_train = load_train_data(train_path)
    X_test_normal, X_test_anomaly = load_test_data(test_path)

    print("Train shape:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    # Global normalization from TRAIN only
    mean = X_train.mean()
    std = X_train.std() + 1e-8

    X_train = (X_train - mean) / std
    X_test_normal = (X_test_normal - mean) / std
    X_test_anomaly = (X_test_anomaly - mean) / std

    # Train model
    print("Training model...")
    model = train_model(X_train)

    # Evaluate
    print("Evaluating...")
    normal_errors = compute_error(model, X_test_normal)
    anomaly_errors = compute_error(model, X_test_anomaly)

    # Threshold
    threshold = compute_threshold(normal_errors)
    print("Threshold:", threshold)

    # Predictions
    pred_normal = [e > threshold for e in normal_errors]
    pred_anomaly = [e > threshold for e in anomaly_errors]

    # Metrics
    false_positive_rate = sum(pred_normal) / len(pred_normal)
    true_positive_rate = sum(pred_anomaly) / len(pred_anomaly)

    print("False Positive Rate:", false_positive_rate)
    print("True Positive Rate (Recall):", true_positive_rate)

    # Plot
    plt.hist(normal_errors, bins=50, alpha=0.5, label="Normal")
    plt.hist(anomaly_errors, bins=50, alpha=0.5, label="Anomaly")
    plt.legend()
    plt.title("Reconstruction Error")
    plt.show()

    print("Done!")

if __name__ == "__main__":
    main()