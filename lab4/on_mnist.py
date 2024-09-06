import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from model import  ResNet
from trainer import Trainer, Dataset
from optim import MomentumSGD


def load_cifar10_data():
    # Define the path to the CIFAR-10 dataset directory
    cifar10_dir = 'datasets/cifar-10-batches-py'

    # Load training data
    train_images, train_labels = [], []
    for i in range(1, 6):
        with open(os.path.join(cifar10_dir, f'data_batch_{i}'), 'rb') as file:
            data_batch = pickle.load(file, encoding='bytes')
            train_images.append(data_batch[b'data'])
            train_labels += data_batch[b'labels']
    train_images = np.concatenate(train_images, axis=0)
    train_images = train_images.reshape(-1, 3, 32, 32)
    train_labels = np.array(train_labels)

    # Load test data
    with open(os.path.join(cifar10_dir, 'test_batch'), 'rb') as file:
        test_batch = pickle.load(file, encoding='bytes')
        test_images = test_batch[b'data']
        test_images = test_images.reshape(-1, 3, 32, 32)
        test_labels = np.array(test_batch[b'labels'])

    return train_images, train_labels, test_images, test_labels


# Load CIFAR-10 data
train_X, train_y, test_X, test_y = load_cifar10_data()

# Print shapes of the loaded data
print("Train images shape:", train_X.shape)
print("Train labels shape:", train_y.shape)
print("Test images shape:", test_X.shape)
print("Test labels shape:", test_y.shape)

train_size = 50000
data_size = 37
model = ResNet(wd=0.000)
dataset = Dataset(train_X[:train_size], train_y[:train_size], test_X[:data_size], test_y[:data_size])
trainer = Trainer(model, dataset, MomentumSGD(), learning_rate=0.0001, num_epochs=3, batch_size=32)

loss_history, train_history, val_history = trainer.fit()
# Best result: val acc - 0.5135

plt.plot(train_history)
plt.plot(val_history)
plt.show()