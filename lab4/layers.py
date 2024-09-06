import numpy as np
import math



def softmax_with_cross_entropy(predictions, target_index):
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)

    batch_size = predictions.shape[0]

    # Shift predictions by subtracting max value for numerical stability
    predictions_shifted = predictions - np.max(predictions, axis=1, keepdims=True)

    # Compute softmax probabilities
    exp_scores = np.exp(predictions_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute cross-entropy loss
    correct_log_probs = -np.log(probs[np.arange(batch_size), target_index])
    loss = np.sum(correct_log_probs) / batch_size

    # Compute gradient
    dprediction = probs
    dprediction[np.arange(batch_size), target_index] -= 1
    dprediction /= batch_size

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X > 0)
        return np.maximum(0, X)

    def backward(self, d_out):
        return d_out * self.mask

    def params(self):
        # No parameters for ReLU
        return {}


class BatchNormLayer:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Param(np.ones((1, num_features, 1, 1)))
        self.beta = Param(np.zeros((1, num_features, 1, 1)))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, X, training=True):
        self.X = X
        if training:
            batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)
            self.normalized_X = (X - batch_mean) / np.sqrt(batch_var + self.eps)
            out = self.gamma.value * self.normalized_X + self.beta.value

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            normalized_X = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma.value * normalized_X + self.beta.value

        return out

    def backward(self, d_out):
        N, C, H, W = d_out.shape

        d_normalized_X = d_out * self.gamma.value
        d_var = np.sum(d_normalized_X * (self.X - self.running_mean) * -0.5 * (self.running_var + self.eps) ** (-1.5),
                       axis=(0, 2, 3), keepdims=True)
        d_mean = np.sum(d_normalized_X * -1 / np.sqrt(self.running_var + self.eps), axis=(0, 2, 3),
                        keepdims=True) + d_var * np.sum(-2 * (self.X - self.running_mean), axis=(0, 2, 3),
                                                        keepdims=True) / (N * H * W)

        dX = d_normalized_X / np.sqrt(self.running_var + self.eps) + d_var * 2 * (self.X - self.running_mean) / (
                    N * H * W) + d_mean / (N * H * W)
        self.gamma.grad = np.sum(d_out * self.normalized_X, axis=(0, 2, 3), keepdims=True)
        self.beta.grad = np.sum(d_out, axis=(0, 2, 3), keepdims=True)

        return dX

    def params(self):
        return {
            'gamma': self.gamma,
            'beta': self.beta,
        }

class SWISHLayer:
    def __init__(self):
        # No parameters to initialize for SWISH
        self.X = None

    def forward(self, X):
        """
        Forward pass of the SWISH activation function.

        SWISH(x) = x * sigmoid(x)

        Parameters:
        X (numpy array): Input data

        Returns:
        numpy array: Activated data
        """
        self.X = X
        self.sigmoid_X = 1 / (1 + np.exp(-X))
        return X * self.sigmoid_X

    def backward(self, d_out):
        """
        Backward pass of the SWISH activation function.

        Parameters:
        d_out (numpy array): Gradient of the loss with respect to the output of the layer

        Returns:
        numpy array: Gradient of the loss with respect to the input of the layer
        """
        sigmoid_derivative = self.sigmoid_X * (1 - self.sigmoid_X)
        dX = d_out * (self.sigmoid_X + self.X * sigmoid_derivative)
        return dX

    def params(self):
        """
        Returns an empty dictionary since SWISH has no learnable parameters.

        Returns:
        dict: Empty dictionary
        """
        return {}

class FullyConnectedLayer:
    def __init__(self, n_input, n_output, reg_coef=0.001):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.reg_coef = reg_coef
        self.X = None

    def forward(self, X):
        """
        Performs forward pass and returns output of the layer.
        """
        self.X = X
        output = np.dot(X, self.W.value) + self.B.value
        return output

    def backward(self, d_out):
        """
        Performs backward pass and computes the gradient
        of the loss with respect to the input.
        """
        # Gradient with respect to X
        d_input = np.dot(d_out, self.W.value.T)

        # Gradient with respect to W
        self.W.grad = np.dot(self.X.T, d_out) + self.reg_coef * self.W.value

        # Gradient with respect to B
        self.B.grad = np.sum(d_out, axis=0, keepdims=True) + self.reg_coef * self.B.value

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvImToColl:
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, reg_coef=0.001):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kh = kernel_size
        self.kw = kernel_size
        self.s = stride
        self.p = padding
        bound = 1 / math.sqrt(self.kh * self.kw)
        self.W = Param(np.random.uniform(-bound, bound, size=(self.out_ch, self.in_ch, self.kh, self.kw)))
        self.b = Param(np.random.uniform(-bound, bound, size=(self.out_ch)))
        self.cache = None
        self.reg = reg_coef

    @staticmethod
    def im2col(image, kernel_size, stride, padding):
        batch_size, num_channels, image_height, image_width = image.shape
        kh, kw = kernel_size

        # Add padding to the image
        padded_image = np.pad(image, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        patches = []
        for k in range(batch_size):
            for i in range(0, image_height + 2 * padding - kh + 1, stride):
                for j in range(0, image_width + 2 * padding - kw + 1, stride):
                    patch = padded_image[k, :, i:i + kh, j:j + kw]
                    patches.append(patch.reshape(-1))

        return np.array(patches)

    @staticmethod
    def col2im(patches, image_shape, kernel_size, stride, padding):
        batch_size, num_channels, image_height, image_width = image_shape
        kh, kw = kernel_size

        # Initialize arrays for the reconstructed image and the count for averaging
        padded_height = image_height + 2 * padding
        padded_width = image_width + 2 * padding
        image = np.zeros((batch_size, num_channels, padded_height, padded_width))
        count = np.zeros((batch_size, num_channels, padded_height, padded_width))

        patch_idx = 0
        for k in range(batch_size):
            for i in range(0, padded_height - kh + 1, stride):
                for j in range(0, padded_width - kw + 1, stride):
                    patch = patches[patch_idx]
                    patch = patch.reshape(num_channels, kh, kw)
                    image[k, :, i:i + kh, j:j + kw] += patch
                    count[k, :, i:i + kh, j:j + kw] += 1
                    patch_idx += 1

        # Remove padding and average overlapping regions
        if padding > 0:
            image = image[:, :, padding:-padding, padding:-padding]
            count = count[:, :, padding:-padding, padding:-padding]

        count[count == 0] = 1
        image /= count

        return image

    def forward(self, x):
        patches = ConvImToColl.im2col(x, (self.kh, self.kw), self.s, self.p)
        patches_reshaped = patches.reshape(patches.shape[0], -1)

        W_reshaped = self.W.value.reshape(self.out_ch, -1).T
        results = np.dot(patches_reshaped, W_reshaped) + self.b.value

        result_height = (x.shape[2] + 2 * self.p - self.kh) // self.s + 1
        result_width = (x.shape[3] + 2 * self.p - self.kw) // self.s + 1
        results = results.reshape(x.shape[0], result_height, result_width, self.out_ch)
        results = results.transpose(0, 3, 1, 2)

        self.cache = (x, patches, patches_reshaped, W_reshaped)

        return results

    def backward(self, dout):
        x, patches, patches_reshaped, W_reshaped = self.cache

        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_ch)

        dW = np.dot(dout_reshaped.T, patches_reshaped)
        dW = dW.reshape(self.out_ch, self.in_ch, self.kh, self.kw)

        db = np.sum(dout_reshaped, axis=0)

        dpatches = np.dot(dout_reshaped, W_reshaped.T)
        dpatches = dpatches.reshape(patches.shape)

        dx = ConvImToColl.col2im(dpatches, x.shape, (self.kh, self.kw), self.s, self.p)
        dW += self.W.value * self.reg
        self.W.grad = dW
        self.b.grad = db

        return dx

    def params(self):
        return {'W': self.W, 'B': self.b}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, input):
        input = input.transpose(0, 2, 3, 1)
        self.input = input

        batch_size, height, width, channels = input.shape
        assert channels == self.in_channels

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        padded_X = np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                          mode='constant')

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for h in range(out_height):
            for w in range(out_width):
                window = padded_X[:, h:h + self.filter_size, w:w + self.filter_size, :]
                out[:, h, w, :] = np.sum(window[:, :, :, :, None] * self.W.value[None, :, :, :, :], axis=(1, 2, 3))

        out += self.B.value
        out = out.transpose(0, 3, 1, 2)
        return out

    def backward(self, d_out):
        d_out = d_out.transpose(0, 2, 3, 1)
        batch_size, height, width, channels = self.input.shape
        _, out_height, out_width, out_channels = d_out.shape

        padded_X = np.pad(self.input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                          mode='constant')
        padded_d_in = np.zeros_like(padded_X)

        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

        for h in range(out_height):
            for w in range(out_width):
                window = padded_X[:, h:h + self.filter_size, w:w + self.filter_size, :]
                for c in range(self.out_channels):
                    self.W.grad[:, :, :, c] += np.sum(window * d_out[:, h, w, c][:, None, None, None], axis=0)
                for n in range(batch_size):
                    for c in range(self.out_channels):
                        padded_d_in[n, h:h + self.filter_size, w:w + self.filter_size, :] += self.W.value[:, :, :, c] * \
                                                                                           d_out[n, h, w, c]

        self.B.grad = np.sum(d_out, axis=(0, 1, 2))

        if self.padding > 0:
            d_in = padded_d_in[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            d_in = padded_d_in
        d_in = d_in.transpose(0, 3, 1, 2)
        return d_in

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):

        batch_size, channels, height, width = X.shape
        self.X = X
        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        # Initialize output tensor
        output = np.zeros((batch_size, channels, out_height, out_width))

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        window = X[b, c, h_start:h_end, w_start:w_end,]
                        output[b, c, i, j] = np.max(window)

        return output

    def backward(self, d_out):
        batch_size, channels, height, width = self.X.shape

        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        # Initialize gradient tensor for the input
        d_input = np.zeros_like(self.X)

        # Perform backpropagation
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        window = self.X[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        mask = (window == max_val)

                        d_input[b, c, h_start:h_end, w_start:w_end] += d_out[b, c, i, j] * mask

        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape  # Save the original shape for the backward pass
        batch_size, channels, height, width = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}