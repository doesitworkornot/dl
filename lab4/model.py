import numpy as np

from layers import (
    FullyConnectedLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, ConvImToColl, SWISHLayer, BatchNormLayer
    )


class BasicBlock:
    def __init__(self, in_channels, out_channels, stride=1, wd=0.0001):
        self.stride = stride
        self.conv1 = ConvImToColl(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, reg_coef=wd)
        self.bn1 = BatchNormLayer(out_channels)
        self.swish1 = SWISHLayer()
        self.conv2 = ConvImToColl(out_channels, out_channels, kernel_size=3, stride=1, padding=1, reg_coef=wd)
        self.bn2 = BatchNormLayer(out_channels)
        self.swish2 = SWISHLayer()

        if in_channels != out_channels or stride != 1:
            self.shortcut = ConvImToColl(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, reg_coef=wd)
            self.bn_shortcut = BatchNormLayer(out_channels)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.swish1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.shortcut is not None:
            identity = self.shortcut.forward(x)
            identity = self.bn_shortcut.forward(identity)

        out += identity
        out = self.swish2.forward(out)

        return out

    def backward(self, d_out):
        if self.shortcut is not None:
            d_out_shortcut = self.bn_shortcut.backward(d_out)
            d_out_shortcut = self.shortcut.backward(d_out_shortcut)
        else:
            d_out_shortcut = d_out

        d_out = self.swish2.backward(d_out)

        d_out = self.bn2.backward(d_out)
        d_out = self.conv2.backward(d_out)

        d_out = self.swish1.backward(d_out)
        d_out = self.bn1.backward(d_out)
        d_out = self.conv1.backward(d_out)

        if self.shortcut is not None:
            d_out += d_out_shortcut

        return d_out

    def params(self):
        params = {**self.conv1.params(), **self.bn1.params(), **self.conv2.params(), **self.bn2.params()}
        if self.shortcut is not None:
            params.update(self.shortcut.params())
            params.update(self.bn_shortcut.params())
        return params


class ResNet:
    def __init__(self, wd=0.0001):
        self.layers = [
            ConvImToColl(3, 16, kernel_size=3, stride=1, padding=1, reg_coef=wd),
            BatchNormLayer(16),
            SWISHLayer(),
            self._make_layer(16, 16, stride=1, num_blocks=2, wd=wd),
            self._make_layer(16, 32, stride=2, num_blocks=2, wd=wd),
            self._make_layer(32, 64, stride=2, num_blocks=2, wd=wd),
            MaxPoolingLayer(2, 2),
            Flattener(),
            FullyConnectedLayer(64 * 4 * 4, 10)  # Assuming 10 classes for CIFAR-10
        ]
        self.reg = wd

    def _make_layer(self, in_channels, out_channels, stride, num_blocks, wd):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, wd=wd))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, wd=wd))
        return layers

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, list):
                for block in layer:
                    x = block.forward(x)
            else:
                x = layer.forward(x)
        self.predictions = x
        return x

    def backward(self, d_out):
        d_out = self.layers[-1].backward(d_out)  # FullyConnectedLayer
        d_out = self.layers[-2].backward(d_out)  # Flattener
        for layer in reversed(self.layers[:-2]):
            if isinstance(layer, list):
                for block in reversed(layer):
                    d_out = block.backward(d_out)
            else:
                d_out = layer.backward(d_out)
        return d_out

    def compute_loss_and_gradients(self, X, y):
        self.zero_grad()
        predictions = self.forward(X)
        loss, dpred = softmax_with_cross_entropy(predictions, y)
        self.backward(dpred)
        return loss

    def predict(self, X):
        pred = self.forward(X)
        y_pred = np.argmax(pred, axis=1)
        return y_pred

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, list):
                for block in layer:
                    for name, param in block.params().items():
                        if param.grad is not None:
                            param.grad.fill(0)
            else:
                for name, param in layer.params().items():
                    if param.grad is not None:
                        param.grad.fill(0)

    def params(self):
        params = {}
        for layer in self.layers:
            if isinstance(layer, list):
                for block in layer:
                    params.update(block.params())
            else:
                params.update(layer.params())
        return params

class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # Initialize layers
        self.conv1 = ConvImToColl(input_shape[2], conv1_channels, 3, padding=1)
        self.bn1 = BatchNormLayer(conv1_channels)
        self.relu1 = SWISHLayer()
        self.maxpool1 = MaxPoolingLayer(4, 4)
        self.conv2 = ConvImToColl(conv1_channels, conv2_channels, 3, padding=1)
        self.bn2 = BatchNormLayer(conv2_channels)
        self.relu2 = SWISHLayer()
        self.maxpool2 = MaxPoolingLayer(4, 4)
        self.flatten = Flattener()
        fc_input_dim = (input_shape[0] // 16) * (input_shape[1] // 16) * conv2_channels
        self.fc = FullyConnectedLayer(fc_input_dim, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Clear parameter gradients
        for param_name, param in self.params().items():
            param.grad = np.zeros_like(param.value)

        # Forward pass
        conv1_out = self.conv1.forward(X)
        bn1_out = self.bn1.forward(conv1_out)
        relu1_out = self.relu1.forward(bn1_out)
        maxpool1_out = self.maxpool1.forward(relu1_out)
        conv2_out = self.conv2.forward(maxpool1_out)
        bn2_out = self.bn2.forward(conv2_out)
        relu2_out = self.relu2.forward(bn2_out)
        maxpool2_out = self.maxpool2.forward(relu2_out)
        flattened = self.flatten.forward(maxpool2_out)
        fc_out = self.fc.forward(flattened)

        # Compute loss and gradients
        loss, d_fc_out = softmax_with_cross_entropy(fc_out, y)
        d_flattened = self.fc.backward(d_fc_out)
        d_maxpool2_out = self.flatten.backward(d_flattened)
        d_relu2_out = self.maxpool2.backward(d_maxpool2_out)
        d_bn2_out = self.relu2.backward(d_relu2_out)
        d_conv2_out = self.bn2.backward(d_bn2_out)
        d_maxpool1_out = self.conv2.backward(d_conv2_out)
        d_relu1_out = self.maxpool1.backward(d_maxpool1_out)
        d_bn1_out = self.relu1.backward(d_relu1_out)
        d_conv1_out = self.bn1.backward(d_bn1_out)
        _ = self.conv1.backward(d_conv1_out)  # No need to store the gradients

        return loss

    def predict(self, X):
        """
        Produces class predictions for a batch of inputs

        Arguments:
          X, np array (batch_size, height, width, input_features) - input data

        Returns:
          y_pred, np array of int (batch_size) - predicted classes
        """
        # Forward pass
        conv1_out = self.conv1.forward(X)
        bn1_out = self.bn1.forward(conv1_out, False)
        relu1_out = self.relu1.forward(bn1_out)
        maxpool1_out = self.maxpool1.forward(relu1_out)
        conv2_out = self.conv2.forward(maxpool1_out)
        bn2_out = self.bn2.forward(conv2_out, False)
        relu2_out = self.relu2.forward(bn2_out)
        maxpool2_out = self.maxpool2.forward(relu2_out)
        flattened = self.flatten.forward(maxpool2_out)
        fc_out = self.fc.forward(flattened)

        # Predict classes
        y_pred = np.argmax(fc_out, axis=1)

        return y_pred

    def params(self):
        """
        Returns parameters of your network
        """
        result = {}

        # Aggregate all the parameters from all the layers
        # which have parameters
        result.update(self.conv1.params())
        result.update(self.conv2.params())
        result.update(self.fc.params())

        return result
