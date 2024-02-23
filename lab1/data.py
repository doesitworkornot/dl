import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms


class SimpsonDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = ImageFolder(data_path, transform=self.transform())
        class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        self.create_splits()

    def transform(self):
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform
    def create_splits(self):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split1 = int(0.8 * dataset_size)  # 80% на обучение
        split2 = int(0.1 * dataset_size)  # 10% на валидацию
        torch.manual_seed(13)  # для воспроизводимости
        indices = torch.randperm(len(indices))
        self.train_indices, self.val_indices, self.test_indices = indices[:split1], indices[split1:split1+split2], indices[split1+split2:]

        # Убедитесь, что в каждом классе есть хотя бы один представитель в тестовой выборке
        for class_idx in self.dataset.classes:
            if class_idx not in [self.dataset.targets[i] for i in self.test_indices]:
                # Найти изображение этого класса в обучающей выборке и добавить его в тестовую выборку
                for idx in self.train_indices:
                    if self.dataset.targets[idx] == class_idx:
                        self.test_indices.append(idx)
                        self.train_indices.remove(idx)
                        break

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

    def get_loaders(self, batch_size=64):
        train_sampler = SubsetRandomSampler(self.train_indices)
        val_sampler = SubsetRandomSampler(self.val_indices)
        test_sampler = SubsetRandomSampler(self.test_indices)

        train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler)

        return train_loader, val_loader, test_loader


def data_prepare(path, batch_size):
    # Create your dataset
    dataset = SimpsonDataset(path)

    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size)

    class_names = dataset.idx_to_class

    return train_loader, val_loader, test_loader, class_names
