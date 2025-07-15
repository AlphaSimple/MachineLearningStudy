import numpy as np

def accuracy(y_pred, y_true):
    return np.mean(np.sign(y_pred) == y_true)

def square_loss(y_pred, y_true):
    return 0.5 * (y_pred - y_true)**2

def train_sgd(model, X_train, y_train, X_test, y_test, gamma0, d, max_epochs=20):
    N = X_train.shape[0]
    t = 0
    train_losses = []

    for epoch in range(max_epochs):
        np.random.seed(epoch+1)
        indices = np.random.permutation(N)
        for i in indices:
            x, y = X_train[i], y_train[i]
            model.forward(x)
            grads = model.backward(y)

            gamma_t = gamma0 / (1 + (gamma0 / d) * t)
            model.update_weights(grads, gamma_t)
            t += 1

        y_train_pred = np.array([model.forward(x) for x in X_train])
        loss = np.mean(square_loss(y_train_pred, y_train))
        train_losses.append(loss)

    y_train_pred = np.sign(np.array([model.forward(x) for x in X_train]))
    y_test_pred = np.sign(np.array([model.forward(x) for x in X_test]))

    train_error = 1 - accuracy(y_train_pred, y_train)
    test_error = 1 - accuracy(y_test_pred, y_test)

    return train_error, test_error, train_losses
