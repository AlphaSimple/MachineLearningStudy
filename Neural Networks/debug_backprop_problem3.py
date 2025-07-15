import numpy as np
from model import NeuralNet

# Given input and label
x = np.array([1, 1])
y_true = 1

# Initialize model with width 2 (as in paper problem 3)
model = NeuralNet(input_dim=3, hidden_width=2)

# Manually set weights to match the problem table
model.w1 = np.array([
    [-1, 1],   # Bias weights
    [-2, 2],   # x1 to h1, h2
    [-3, 3],   # x2 to h1, h2
])

model.w2 = np.array([
    [-1, 1],    # Bias to h1, h2
    [-2, 2],    # a1^1 to h1, h2
    [-3, 3],    # a2^1 to h1, h2
])

model.w3 = np.array([
    -1,    # Bias
    2,     # a1^2
    -1.5   # a2^2
])

# Forward pass
y_pred = model.forward(x)
print(f"Forward output y = {y_pred:.4f} (expected: -2.437)")

# Backward pass
grads = model.backward(y_true)
gw1, gw2, gw3 = grads

print("\nGradient w.r.t. weights in Layer 3 (w3):")
for i, val in enumerate(gw3):
    print(f"∂L/∂w3[{i}] = {val:.6f}")

print("\nGradient w.r.t. weights in Layer 2 (w2):")
for i in range(gw2.shape[0]):
    for j in range(gw2.shape[1]):
        print(f"∂L/∂w2[{i},{j}] = {gw2[i,j]:.6f}")

print("\nGradient w.r.t. weights in Layer 1 (w1):")
for i in range(gw1.shape[0]):
    for j in range(gw1.shape[1]):
        print(f"∂L/∂w1[{i},{j}] = {gw1[i,j]:.6f}")
