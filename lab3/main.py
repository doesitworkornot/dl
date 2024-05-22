import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import copy

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn.functional as F

import unet
from iou import calculate_iou


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images_directory, masks_directory, transform=None):
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform

        self.images_filenames = sorted(os.listdir(self.images_directory))

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_directory, image_filename), cv2.IMREAD_COLOR)[:, :, 0:1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = np.transpose(mask, (2, 0, 1))

            # mask = torch.from_numpy(mask)
        return image, mask


def visualize_augmentations(dataset, idx=0, samples=10):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        mask = np.transpose(mask, (1,2,0))
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()


def main():

    ds_images_path = 'train/images'
    ds_masks_path = 'train/labels'
    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=300, min_width=300),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(256, 256),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            # A.Blur(p=0.2),
            # A.GaussNoise(0.3, 0.2, p=0.2),
            # A.RandomBrightnessContrast(p=0.5),
            # A.Perspective(p=0.5),
            ToTensorV2(),
        ]
    )
    ds_train = MyDataset(ds_images_path, ds_masks_path, transform=train_transform)
    visualize_augmentations(ds_train)

    batch_size = 8
    train_loader = torch.utils.data.DataLoader(
        ds_train, shuffle=True,
        batch_size=batch_size, num_workers=1, drop_last=True
    )

    device = 'cuda'
    unet_model = unet.Unet(backbone_name='densenet169')
    unet_model = unet_model.to(device)
    max_lr = 0.01
    epoch = 500
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(unet_model.parameters(), lr=max_lr, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epoch, steps_per_epoch=len(train_loader))

    max_score = 10000
    epochs = epoch
    for epoch in range(epochs):
        loss_val = 0
        for sample in train_loader:
            img, mask = sample
            img = img.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            pred = unet_model(img)
            loss = loss_fn(pred, mask)

            loss.backward()
            loss_item = loss.item()
            loss_val += loss_item
            iou = calculate_iou(mask.cpu().detach().numpy(), pred.cpu().detach().numpy())

            optimizer.step()

        if max_score > loss_val:
            max_score = loss_val
            torch.save(unet_model, 'best_model169.pth')
            print('Model saved!')

        scheduler.step()
        print(f'{epoch+1}th epoch:')
        print(f'IoU: {iou}')
        print(f'{loss_val / len(train_loader)}\t lr: {scheduler.get_last_lr()[0]}')

    img = ds_train[10][0].unsqueeze(0)
    pred = unet_model(img.cuda())
    pred = F.sigmoid(pred.detach()).cpu().numpy()[0].transpose(1, 2, 0)

    img_np = img.detach().cpu().numpy()[0].transpose(1, 2, 0)
    plt.imshow(img_np)
    plt.show()

    plt.imshow(pred)
    plt.show()


if __name__ == '__main__':
    main()
