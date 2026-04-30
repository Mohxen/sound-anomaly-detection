import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.model import CNNAutoencoder
from src.config import EPOCHS, LR, BATCH_SIZE, RANDOM_SEED

def train_model(X_train):
    torch.manual_seed(RANDOM_SEED)
    X = torch.tensor(X_train, dtype=torch.float32)
    X = X.unsqueeze(1)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    model = CNNAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        total_items = 0

        for (x,) in loader:
            output = model(x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            total_items += len(x)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss {total_loss / total_items}")

    return model
