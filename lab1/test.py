import argparse

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
import numpy as np


class SimpsonDataset(Dataset):
    def __init__(self, root_dir):
        self.dataset = ImageFolder(root=root_dir)

    def get_classes(self):
        return self.dataset.classes

    def get_idx(self):
        return self.dataset.class_to_idx


class TestDataset(Dataset):
    def __init__(self, root_dir):
        transform = transforms.Compose([
            transforms.Resize((280, 280)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ])

        self.dataset = ImageFolder(root=root_dir, transform=transform)

    def get_test_loader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def get_classes(self):
        return self.dataset.classes


def test_step(model, test_loader, device, idx_to_name_test, name_to_idx):
    model.eval()
    y_pred = []
    y_test = []
    for img, labels in test_loader:
        model.eval()
        img, labels = img.to(device), labels.to(device)
        target = model(img)
        y_pred.append(target.detach().cpu().numpy())
        idx = labels.detach().cpu().numpy()
        np_idx_to = np.array(idx_to_name_test)
        to_name = np_idx_to[idx]
        idx_array = np.array([name_to_idx[name] for name in to_name])
        y_test.append(idx_array)
    y_test = np.concatenate(y_test)
    y_pred = np.argmax(np.concatenate(y_pred), axis=1)
    return y_pred, y_test


def main(arg):
    if not arg.network:
        print('type -help to see usage')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torchvision.models.resnet50().to(device)
        bs = 50
        model.load_state_dict(torch.load(arg.network))
        print('Loaded pretrained model')

        train_dataset = SimpsonDataset('dataset/simpsons_dataset')
        idx_to_name_train = train_dataset.get_classes()
        name_to_idx = train_dataset.get_idx()

        test_dataset = TestDataset('hm_test/new')
        test_loader = test_dataset.get_test_loader(bs)
        idx_to_name_test = test_dataset.get_classes()

        y_pred, y_test = test_step(model, test_loader, device, idx_to_name_test, name_to_idx)

        unique_numbers = sorted(set(y_test))
        np_idx_to = np.array(idx_to_name_train)
        to_name = np_idx_to[unique_numbers]

        print('\n', classification_report(y_test, y_pred, target_names=to_name, zero_division=0))
        print(y_test)
        print(y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default=None, help='Path to your pretrained nn')
    args = parser.parse_args()

    main(args)
