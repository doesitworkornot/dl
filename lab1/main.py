import argparse
import time

from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

import network
import data


def main(arg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torchvision.models.resnet50().to(device)
    bs = 50

    print('Started to load data')
    tic = time.perf_counter()
    train_loader, val_loader, test_loader = data.data_prepare('dataset/simpsons_dataset', batch_size=bs)
    toc = time.perf_counter()
    print(f"It took {toc - tic:0.4f} seconds")

    if arg.network:
        model.load_state_dict(torch.load(arg.network))
        print('Loaded pretrained model')
    else:
        max_lr = 0.01
        epoch = 100
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=0.001)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epoch, steps_per_epoch=len(train_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epoch/5))

        writer = SummaryWriter('runs/lr01_ep100_wc001_augs_cos')
        print('Created a model')

        tic = time.perf_counter()
        print('Starting to educate')
        for i in range(epoch):
            subtic = time.perf_counter()
            train_loss = network.train_step(model, train_loader, optimizer, scheduler, criterion)
            valid_loss, valid_acc = network.valid_step(model, val_loader, criterion)

            writer.add_scalar('Loss/train', train_loss, i)
            writer.add_scalar('Loss/val', valid_loss, i)
            writer.add_scalar('Acc/val', valid_acc, i)
            writer.add_scalar('lr/CosineAnnealingLR', scheduler.get_last_lr()[0], i)
            print(f'train loss: {train_loss}, val loss: {valid_loss}, val acc: {valid_acc}')
            print(scheduler.get_last_lr())

            subtoc = time.perf_counter()
            print(f"{i}th epoch took {subtoc - subtic:0.1f} seconds")
        toc = time.perf_counter()
        print(f"\nAll education took {toc - tic:0.1f} seconds")
        torch.save(model.state_dict(), 'model.pth')
        writer.close()

    y_pred, y_test = network.test_step(model, test_loader, device)
    print('\n', classification_report(y_test, y_pred, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default=None, help='Path to your pretrained nn')
    args = parser.parse_args()

    main(args)
