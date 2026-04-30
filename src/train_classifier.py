import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import BATCH_SIZE, EPOCHS, LR, RANDOM_SEED
from src.model import CNNBinaryClassifier


def train_classifier(X_train, y_train):
    torch.manual_seed(RANDOM_SEED)
    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y_train, dtype=torch.float32)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    model = CNNBinaryClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    positives = y.sum()
    negatives = len(y) - positives
    pos_weight = negatives / positives.clamp_min(1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_items = 0

        for x, target in loader:
            logits = model(x)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            total_items += len(x)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss {total_loss / total_items}")

    return model
