import numpy as np
import matplotlib.pyplot as plt
from dataset import load_svhn, random_split_train_val
from gradient_check import check_layer_gradient, check_layer_param_gradient, check_model_gradient
from layers import FullyConnectedLayer, ReLULayer, ConvolutionalLayer, MaxPoolingLayer, Flattener
from model import ConvNet, ResNet
from trainer import Trainer, Dataset
from optim import SGD, MomentumSGD
from mnist import MNIST
from metrics import multiclass_accuracy


def prepare_for_neural_network(train_X, test_X):
    train_X = train_X.astype(np.float) / 255.0
    test_X = test_X.astype(np.float) / 255.0

    # Subtract mean
    mean_image = np.mean(train_X, axis=0)
    train_X -= mean_image
    test_X -= mean_image

    return train_X, test_X


train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)
train_X, test_X = prepare_for_neural_network(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val=1000)


# No need to use L2 regularization
train_X = train_X.transpose(0, 3, 1, 2)
val_X = val_X.transpose(0, 3, 1, 2)

train_size = 10
data_size = 20
model = ResNet()
# model = ConvNet(input_shape=(32, 32, 3), n_output_classes=10, conv1_channels=4, conv2_channels=8)
dataset = Dataset(train_X[:train_size], train_y[:train_size], val_X[:data_size], val_y[:data_size])
# TODO: Change any hyperparamers or optimizators to reach 1.0 training accuracy in 50 epochs or less
# Hint: If you have hard time finding the right parameters manually, try grid search or random search!
trainer = Trainer(model, dataset, MomentumSGD(), learning_rate=1e-5, num_epochs=100, batch_size=20)

loss_history, train_history, val_history = trainer.fit()

plt.plot(train_history)
plt.plot(val_history)
plt.show()
