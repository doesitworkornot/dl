from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision import transforms


class SimpsonDataset(Dataset):
    def __init__(self, root_dir):
        transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        self.dataset = ImageFolder(root=root_dir, transform=transform)

        train_ratio = 0.8
        test_ratio = 0.1
        val_ratio = 0.1
        train_data, test_data = train_test_split(self.dataset, test_size=1 - train_ratio, random_state=42)
        test_data, val_data = train_test_split(test_data, test_size=val_ratio / (test_ratio + val_ratio),
                                               random_state=42)

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def get_train_loader(self, batch_size):
        return DataLoader(self.train_data, batch_size=batch_size, shuffle=True)

    def get_test_loader(self, batch_size):
        return DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

    def get_val_loader(self, batch_size):
        return DataLoader(self.val_data, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def data_prepare(path, batch_size):
    dataset = SimpsonDataset(path)

    train_loader = dataset.get_train_loader(batch_size=batch_size)
    test_loader = dataset.get_test_loader(batch_size=batch_size)
    val_loader = dataset.get_val_loader(batch_size=batch_size)

    return train_loader, val_loader, test_loader
