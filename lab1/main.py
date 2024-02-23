import argparse
import time

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch
import torch.nn as nn

import network
import data


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()
    model = network.ConvolutionNetwork(criterion).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print('Started to load data')
    tic = time.perf_counter()
    train_loader, val_loader, test_loader, classes = data.data_prepare('dataset/simpsons_dataset', batch_size=64)
    toc = time.perf_counter()
    print(f"It took {toc - tic:0.4f} seconds")

    if args.network:
        model.load_state_dict(torch.load(args.network))
        print('Loaded pretrained model')
    else:
        print('Created a model')
        epoch = 10
        train_losses = []
        valid_losses = []
        valid_accs = []
        tic = time.perf_counter()
        print('Starting to educate')
        for i in range(epoch):
            print(f'{i}th epoch is going')
            subtic = time.perf_counter()
            train_loss = network.train_step(model, train_loader, optimizer)
            valid_loss, valid_acc = network.valid_step(model, val_loader)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            subtoc = time.perf_counter()
            print(f"{i}th epoch took {subtoc - subtic:0.1f} seconds\n")
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
    print(len(classes.values()), classes.values(), max(y_test), max(y_pred))
    print('\n', classification_report(y_test, y_pred, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default=None, help='Path to your pretrained nn')
    args = parser.parse_args()

    main(args)