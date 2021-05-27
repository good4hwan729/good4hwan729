"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        grad = np.zeros((self.w.shape[0], self.n_class))

        for i in range(len(X_train)):
            wc = X_train[i].dot(self.w)
            wy = wc[y_train[i]]
            for j in range(self.n_class):
                if j == y_train[i]:
                    continue
                if wy - wc[j] - 1 < 0:
                    grad[:,y_train[i]] -= X_train[i]
                    grad[:,j] += X_train[i]

        grad /= len(X_train)
        grad += self.reg_const * self.w

        return grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        batch_size = 1000

        if self.w is None:
            self.w = np.random.randn(X_train.shape[1], self.n_class) * 0.01

        for i in range(self.epochs):
            x = np.random.choice(len(X_train),batch_size)
            grad = self.calc_gradient(X_train[x], y_train[x])

            self.w -= self.alpha * grad


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
        y_test = np.argmax(np.dot(X_test, self.w), axis = 1)
        return y_test
