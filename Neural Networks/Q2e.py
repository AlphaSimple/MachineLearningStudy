import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import load_data
from sklearn.metrics import accuracy_score
import itertools

class CustomNet(nn.Module):
    def __init__(self, input_dim, depth, width, activation_fn, init_type):
        super(CustomNet, self).__init__()
        self.depth = depth
        self.activation_name = activation_fn

        # Choose activation
        if activation_fn == 'tanh':
            self.activation = nn.Tanh()
        elif activation_fn == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation.")

        # Build network
        layers = []
        layers.append(nn.Linear(input_dim, width))
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, 1))
        self.layers = nn.ModuleList(layers)

        # Apply initialization
        self.apply(lambda m: self.init_weights(m, init_type))

    def init_weights(self, m, init_type):
        if isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'he':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.depth - 1:  # Don't apply activation to last layer
                x = self.activation(x)
        return x.squeeze(1)

def train_and_evaluate(X_train, y_train, X_test, y_test, depth, width, activation):
    input_dim = X_train.shape[1]
    init_type = 'xavier' if activation == 'tanh' else 'he'
    model = CustomNet(input_dim, depth, width, activation, init_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare data
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_train = torch.sign(model(X_train_t)).cpu().numpy()
        y_pred_test = torch.sign(model(X_test_t)).cpu().numpy()

    train_error = 1 - accuracy_score(y_train, y_pred_train)
    test_error = 1 - accuracy_score(y_test, y_pred_test)
    return train_error, test_error

def main():
    zip_path = '../datasets/bank-note.zip'
    train_file = 'bank-note/train.csv'
    test_file = 'bank-note/test.csv'
    X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

    activations = ['tanh', 'relu']
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]

    results = []

    for act, d, w in itertools.product(activations, depths, widths):
        print(f"Training: activation={act}, depth={d}, width={w}")
        train_err, test_err = train_and_evaluate(X_train, y_train, X_test, y_test, d, w, act)
        print(f"Train Error: {train_err:.4f}, Test Error: {test_err:.4f}\n")
        results.append((act, d, w, train_err, test_err))

    # Summary table
    print("\nSummary of Results:")
    print("{:<6} {:<6} {:<6} {:<12} {:<12}".format("Act", "Depth", "Width", "Train Error", "Test Error"))
    for act, d, w, tr, te in results:
        print(f"{act:<6} {d:<6} {w:<6} {tr:<12.4f} {te:<12.4f}")

if __name__ == "__main__":
    main()
