import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_concrete_data, split_data
from gradient_descent import batch_gradient_descent, compute_cost

def plot_cost_history(cost_history, learning_rate):
    plt.figure(figsize=(8,5))
    plt.plot(cost_history, label=f'Learning rate: {learning_rate}')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Value vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cost_history_plot_bgd.png')
    plt.show() 

if __name__ == "__main__":
    zip_path = "../datasets/concrete+slump+test.zip"
    # Load data
    df = load_concrete_data(zip_path)
    X_train, y_train, X_test, y_test = split_data(df)

    # Learning rate tuning
    r = 1
    tol = 1e-6
    best_w = None
    best_b = None
    best_r = None
    best_cost_history = None
    best_steps = None
    max_iter = 10000
    while True:
        print(f"Trying learning rate: {r}")
        w, b, cost_history, steps = batch_gradient_descent(X_train, y_train, r, tol=tol, max_iter=max_iter)
        if len(cost_history) < max_iter:  # Converged
            best_w = w
            best_b = b
            best_r = r
            best_cost_history = cost_history
            best_steps = steps
            print(f"Converged in {steps} steps with r={r}")
            break
        else:
            print(f"Did not converge in {len(cost_history)} steps with r={r}")
            r /= 2  # Halve the learning rate for the next iteration

    # Report results
    print("\nFinal learned weight vector:")
    print(best_w)
    print(f"Final bias term: {best_b}")
    print(f"Learning rate used: {best_r}")
    print(f"Number of steps: {best_steps}")
    print(f"Final training cost: {best_cost_history[-1]}")

    # Plot cost function history
    plot_cost_history(best_cost_history, best_r)

    # Compute test cost
    test_cost = compute_cost(X_test, y_test, best_w, best_b)
    print(f"Test set cost of the test data: {test_cost}") 