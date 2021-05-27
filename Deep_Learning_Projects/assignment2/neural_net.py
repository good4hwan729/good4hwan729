"""Neural network model."""

from typing import Sequence

import numpy as np
import scipy.special as scip


def softmax_grad(X, y):
    N = X.shape[0]
    X[np.arange(N),y] -= 1
    X = X/N
    return X

def linear_grad(X, G):
    W_grad = X.T.dot(G)
    b_grad = np.sum(G, axis = 0)
    return W_grad, b_grad

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

        self.m = {}
        self.v = {}
        for k, v in self.params.items():
            self.m[k] = np.zeros_like(v)
            self.v[k] = np.zeros_like(v)

        self.t = 0

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        # TODO: implement me
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me

        return np.maximum(X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output data
        """
        # TODO: implement me
        return np.where(X>0, 1, 0)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        exp = np.exp(X - np.max(X))
        return exp / np.sum(exp)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}


        #Store the input for backpropagation
        self.outputs[0] = X

        for i in range(1, self.num_layers + 1 ):

            #Linear layer
            X = self.linear(self.params["W" + str(i)], X, self.params["b"+ str(i)])
            self.outputs["linear" + str(i)] = X

            if (i < self.num_layers):
                #Relu layer
                X = self.relu(X)
                self.outputs["relu" + str(i)] = X

        #Softmax function works for one at a time
        #Only for the last layer, we do softmax instead of relu
        tmp = X
        for j in range(X.shape[0]):
            tmp[j] = self.softmax(X[j])
        self.outputs["softmax"] = tmp

        #print(tmp, "Forward output")
        return tmp


    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.

        Note: both gradients and loss should include regularization.

        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        #sprint("BACKWARD START")
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        #Call output of forward
        output = self.outputs["softmax"]
        N = np.shape(output)[0]

        #Cross-entropy loss calculation function
        loss = 0
        loss = np.sum(-np.log(output[range(N), y]))
        loss /= N

        #Calculate loss w/ regularization
        for i in range(1, self.num_layers + 1):

            W = self.params["W" + str(i)]
            loss += reg * np.sum(W*W)


        #store input for backwards
        self.outputs["relu" + str(0)] = self.outputs[0]

        for i in range(self.num_layers, 0, -1):

            #Softmax layer
            if (i == self.num_layers):
                output = softmax_grad(output, y)

            #Relu layer
            else:
                X_input = self.outputs["linear" + str(i)]
                output = np.where(X_input < 0, 0, output)

            #Linear and update gradients and output w/ regularization
            X_input = self.outputs["relu" + str(i-1)]
            W_grad, b_grad = linear_grad(X_input, output)
            output =  output.dot(self.params["W" + str(i)].T)

            self.gradients["W" + str(i)] = W_grad + reg * self.params["W" + str(i)] * 2
            #print(self.gradients[W], "thing1")
            self.gradients["b" + str(i)] = b_grad
            #print(self.gradients[W], "thing2")

        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.

        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.

        #SGD
        if opt == "SGD":
            for i in range (1 ,self.num_layers+1):
                self.params["W" + str(i)] -= self.gradients["W" + str(i)] * lr
                self.params["b" + str(i)] -= self.gradients["b" + str(i)] * lr

        else:
            self.t += 1

            for i in self.params.keys():

                self.m[i] = b1 * self.m[i] + (1 - b1) * self.gradients[i]
                self.v[i] = b2 * self.v[i] + (1 - b2) * self.gradients[i]**2

                m_hat = self.m[i] / (1 - b1**self.t)
                v_hat = self.v[i] / (1 - b2**self.t)

                self.params[i] -= lr * (m_hat / np.sqrt(v_hat) + eps)


        pass
