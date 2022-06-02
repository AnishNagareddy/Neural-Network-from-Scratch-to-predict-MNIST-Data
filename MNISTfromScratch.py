from cmath import exp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def init_params():
    # initialize all the weights and biases
    W1 = np.random.rand(10, 785) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


# rectiied linear unit function
def ReLu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return exp(Z) / np.sum(exp(Z))


# forward propogation
def forward_propogation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# one_hot function to classify the number in terms of an array
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# derivative of the Relu funtion
def ReLu_deriv(Z):
    return Z > 0


# backward propogation
def backward_propogation(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLu_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


# updating the parameters based on the propogation terms
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# getting the gradient descnent
def get_pred(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_d(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propogation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propogation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_pred(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propogation(W1, b1, W2, b2, X)
    predictions = get_pred(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)  # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _, m_train = X_train.shape

    W1, b1, W2, b2 = gradient_d(X_train, Y_train, 500, 0.1)

    test_prediction(0, W1, b1, W2, b2)

    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    get_accuracy(dev_predictions, Y_dev)
