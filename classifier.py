import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import get_dataset, get_device

CLASSIFIER_DIR = Path("classifier")
JUDGE_SIZE = 28


class CNN(nn.Module):
    def __init__(self, image_size=16):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv2, self.bn2 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv3, self.bn3 = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        side = image_size // 8  # 3 max-pool
        self.fc1 = nn.Linear(128 * side * side, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return F.relu(self.fc1(x.flatten(1)))

    def forward(self, x):
        return self.fc2(self.dropout(self.features(x)))


def to_judge(x):
    if x.shape[-1] != JUDGE_SIZE:
        x = F.interpolate(x, size=JUDGE_SIZE, mode="bilinear", align_corners=False)
    return x


def classifier_path(image_size):
    return CLASSIFIER_DIR / f"mnist_cnn_{image_size}.pt"


def load_classifier(image_size, device):
    path = classifier_path(image_size)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python classifier.py --image_size {image_size} first")
    clf = CNN(image_size).to(device)
    clf.load_state_dict(torch.load(path, map_location=device))
    return clf.eval()


@torch.no_grad()
def accuracy(clf, loader, device):
    correct = total = 0
    for x, y in loader:
        correct += (clf(x.to(device)).argmax(1).cpu() == y).sum().item()
        total += len(y)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    # Make the judge robust to slight rotations/translations.
    train_tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomAffine(10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.MNIST("data", train=True, download=True, transform=train_tf)
    test = get_dataset(args.image_size, train=False)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    test_loader = DataLoader(test, batch_size=1000)

    clf = CNN(args.image_size).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        clf.train()
        for x, y in train_loader:
            loss = F.cross_entropy(clf(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        clf.eval()
        print(f"Epoch {epoch}/{args.epochs}  test accuracy {accuracy(clf, test_loader, device):.2%}")

    CLASSIFIER_DIR.mkdir(exist_ok=True)
    torch.save(clf.state_dict(), classifier_path(args.image_size))
    print(f"Saved: {classifier_path(args.image_size)}")


if __name__ == "__main__":
    main()
