import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class ConvolutionNetwork(nn.Module):
    def __init__(self, criterion, num_classes: int = 42):
        super().__init__()
        self.criterion = criterion
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def train_step(model, train_dataloader, optimizer) -> float:
    model.train()
    cuda0 = torch.device('cuda:0')
    running_loss = 0.
    for images, labels in train_dataloader:
        optimizer.zero_grad()

        images = images.to(cuda0)
        labels = labels.to(cuda0)

        output = model(images)
        loss = model.criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().numpy()

    with torch.no_grad():
        train_loss = running_loss / len(train_dataloader)

    return train_loss


def valid_step(model, valid_dataloader) -> tuple[float, float]:
    model.eval()
    cuda0 = torch.device('cuda:0')
    correct_total = 0.
    running_loss = 0.
    with torch.no_grad():
        for images, labels in valid_dataloader:
            images = images.to(cuda0)
            labels = labels.to(cuda0)

            output = model(images)
            prediction = output.argmax(dim=1)

            correct_total += prediction.eq(labels.view_as(prediction)).sum().detach().cpu().numpy()
            loss = model.criterion(output, labels)
            running_loss += loss.detach().cpu().numpy()

    valid_loss = running_loss / len(valid_dataloader)
    accuracy = correct_total / len(valid_dataloader.dataset)
    return valid_loss, accuracy


def test_step(model, test_loader, device):
    model.eval()

    y_pred = []
    y_test = []
    for img, labels in test_loader:
        model.eval()
        img, labels = img.to(device), labels.to(device)
        target = model(img)
        y_pred.append(target.detach().cpu().numpy())
        y_test.append(labels.detach().cpu().numpy())
    y_test = np.concatenate(y_test)
    y_pred = np.argmax(np.concatenate(y_pred), axis=1)
    return y_pred, y_test
