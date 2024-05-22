import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from iou import calculate_iou


def visualize_images(simple_img, gt, pred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    simple_img = np.squeeze(simple_img, axis=0)
    gt = np.squeeze(gt, axis=0)
    pred = np.squeeze(pred.detach().numpy(), axis=0)

    gt = np.transpose(gt, (1, 2, 0))
    pred = np.transpose(pred, (1, 2, 0))
    simple_img = np.transpose(simple_img, (1, 2, 0))

    axes[0].imshow(simple_img)
    axes[0].set_title('Simple Image')
    axes[0].axis('off')

    axes[1].imshow(gt)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.show()


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


def e():
    val_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.CenterCrop(256, 256),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    ds_images_path = 'test/images'
    ds_masks_path = 'test/labels'
    device = 'cuda'
    ds_train = MyDataset(ds_images_path, ds_masks_path, transform=val_transform)
    model = torch.load('best_model169.pth')
    unet_model = model.to(device)
    train_loader = torch.utils.data.DataLoader(
        ds_train, shuffle=True,
        batch_size=1, num_workers=1, drop_last=True
    )
    cum_iou = 0
    times = 3
    model.eval()
    for i in range(times):
        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)
            pred = unet_model(img)
            iou = calculate_iou(mask.cpu().detach().numpy(), pred.cpu().detach().numpy())
            cum_iou += iou
            # visualize_images(img.cpu(), mask.cpu(), pred.cpu())
    cum_iou /= (len(train_loader) * times)
    print(f'IoU on test sample is: {cum_iou}')


if __name__ == '__main__':
    e()
