import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from pathlib import Path

from data_loader import get_dataloaders


# Paths
TRAIN_DIR = Path("../data/train")
TEST_DIR = Path("../data/test")
MODEL_PATH = Path("../models/saved_models/model.pth")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # Load data
    train_loader, test_loader, classes = get_dataloaders(TRAIN_DIR, TEST_DIR)

    # Model
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Training
    EPOCHS = 5

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print("Model saved at:", MODEL_PATH)

