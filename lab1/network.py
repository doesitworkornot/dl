import torch
import numpy as np


def train_step(model, train_dataloader, optimizer, scheduler, criterion) -> float:
    model.train()
    cuda0 = torch.device('cuda:0')
    running_loss = 0.
    for images, labels in train_dataloader:
        optimizer.zero_grad()

        images = images.to(cuda0)
        labels = labels.to(cuda0)

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.detach().cpu().numpy()

    with torch.no_grad():
        train_loss = running_loss / len(train_dataloader)

    return train_loss


def valid_step(model, valid_dataloader, criterion) -> tuple[float, float]:
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
            loss = criterion(output, labels)
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
