import numpy as np
import math
import warnings

class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = ReLU()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if (i == 0):
                X, y = self.shuffle(X, y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            # print(loss)
            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)


class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        # Write forward pass here
        self.x = input
        activation = input.dot(self.w) + self.b
        return activation

    def backward(self, gradients):
        # Write backward pass here
        # TODO: See if dot is required or something else!!!
        w1 = self.x.transpose().dot(gradients)
        x1 = gradients.dot(self.w.transpose())
        self.w = self.w - self.lr * w1
        self.b = self.b - self.lr * gradients
        return x1


class Sigmoid:

    def __init__(self):
        None

    def forward(self, inputs):
        # Write forward pass here
        input = self.getSigmoid(inputs)
        self.input = input
        return input

    def backward(self, gradients):
        # Write backward pass here
        gradients = gradients * ((1 - self.input) * self.input)
        return gradients

    def getSigmoid(self, n):
        result = 1.0 / (1.0 + np.exp(-n))
        return result

class ReLU:
    def __init__(self):
        None

    def forward(self, inputs):
        input = self.getReLU(inputs)
        self.input = input
        return input

    def backward(self, gradients):
        gradients = gradients * ((1 - self.input) * self.input)
        return gradients

    def getReLU(self, n):
        result = np.maximum(n, 0)
        return result

