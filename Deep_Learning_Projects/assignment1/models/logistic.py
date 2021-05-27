"""Logistic regression model."""

import numpy as np
import math

class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        sig = 1 / (1 + math.exp(-z))
        return sig

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = np.random.randn(1, X_train.shape[1])

        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                if y_train[j] == 0:
                    self.w -= self.lr * self.sigmoid(np.dot(self.w, X_train[j])) * X_train[j]
                else:
                    self.w += self.lr * self.sigmoid(-1 * np.dot(self.w, X_train[j])) * X_train[j]
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        y_test = []
        for i in range(X_test.shape[0]):
            if np.sign(np.dot(self.w, X_test[i])) == -1:
                y_test.append(0)
            else:
                y_test.append(1)
        return y_test
