import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNet:
    def __init__(self, input_dim, hidden_width):
        self.width = hidden_width
        self.init_weights(input_dim)

    def init_weights(self, input_dim):
        # +1 for bias unit
        np.random.seed(42)
        self.w1 = np.random.randn(input_dim + 1, self.width)
        np.random.seed(43)
        self.w2 = np.random.randn(self.width + 1, self.width)
        np.random.seed(44)
        self.w3 = np.random.randn(self.width + 1)

    def forward(self, x):
        x = np.insert(x, 0, 1)  # Add bias
        self.a0 = x

        self.z1 = self.a0 @ self.w1
        self.a1 = sigmoid(self.z1)
        self.a1 = np.insert(self.a1, 0, 1)  # Bias
        
        self.z2 = self.a1 @ self.w2
        self.a2 = sigmoid(self.z2)
        self.a2 = np.insert(self.a2, 0, 1)
        
        self.z3 = self.a2 @ self.w3
        self.y_hat = self.z3  # No activation in output

        return self.y_hat

    def backward(self, y_true):
        delta3 = self.y_hat - y_true  # dL/dz3

        grad_w3 = delta3 * self.a2

        dz2 = (self.w3[1:] * delta3) * sigmoid_derivative(self.z2)
        grad_w2 = np.outer(self.a1, dz2)

        dz1 = (self.w2[1:, :] @ dz2) * sigmoid_derivative(self.z1)
        grad_w1 = np.outer(self.a0, dz1)

        return grad_w1, grad_w2, grad_w3

    def update_weights(self, grads, lr):
        gw1, gw2, gw3 = grads
        self.w1 -= lr * gw1
        self.w2 -= lr * gw2
        self.w3 -= lr * gw3

    def set_zero_weights(self, input_dim):
        self.w1 = np.zeros((input_dim + 1, self.width))
        self.w2 = np.zeros((self.width + 1, self.width))
        self.w3 = np.zeros(self.width + 1)