import torch
import torch.nn as nn
from src.model import CNNAutoencoder
from src.config import EPOCHS, LR

def train_model(X_train):
    X = torch.tensor(X_train, dtype=torch.float32)

    X = X.unsqueeze(1)

    model = CNNAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0

        for x in X:
            x = x.unsqueeze(0)

            output = model(x)

            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss {total_loss / len(X)}")

    return model