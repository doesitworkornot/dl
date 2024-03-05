import argparse
import time

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torchvision

import network
import data


def main(arg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()
    model = torchvision.models.resnet50().to(device)

    print('Started to load data')
    tic = time.perf_counter()
    train_loader, val_loader, test_loader = data.data_prepare('dataset/simpsons_dataset', batch_size=32)
    toc = time.perf_counter()
    print(f"It took {toc - tic:0.4f} seconds")

    if arg.network:
        model.load_state_dict(torch.load(arg.network))
        print('Loaded pretrained model')
    else:
        max_lr = 0.001
        epoch = 300
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epoch, steps_per_epoch=len(train_loader))
        print('Created a model')
        train_losses = []
        valid_losses = []
        valid_accs = []
        tic = time.perf_counter()
        print('Starting to educate')
        for i in range(epoch):
            subtic = time.perf_counter()
            train_loss = network.train_step(model, train_loader, optimizer, scheduler, criterion)
            valid_loss, valid_acc = network.valid_step(model, val_loader, criterion)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            subtoc = time.perf_counter()
            print(f"{i}th epoch took {subtoc - subtic:0.1f} seconds")
        toc = time.perf_counter()
        print(f"\nAll education took {toc - tic:0.1f} seconds")
        torch.save(model.state_dict(), 'model.pth')

        fig = plt.figure(figsize=(16, 12))
        plt.plot(train_losses[1:], label='train')
        plt.plot(valid_losses[1:], label='valid')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

        fig = plt.figure(figsize=(16, 12))
        plt.plot(valid_accs)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.show()

    y_pred, y_test = network.test_step(model, test_loader, device)
    print('\n', classification_report(y_test, y_pred, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default=None, help='Path to your pretrained nn')
    args = parser.parse_args()

    main(args)
