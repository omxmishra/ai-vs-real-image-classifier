import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data_loader import get_dataloaders


# Paths
TRAIN_DIR = Path("../data/train")
TEST_DIR = Path("../data/test")
MODEL_PATH = Path("../models/saved_models/model.pth")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    # Load data
    _, test_loader, classes = get_dataloaders(TRAIN_DIR, TEST_DIR)

    # Load model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()

    plt.title("Confusion Matrix")
    plt.show()

