import matplotlib.pyplot as plt
from data_loader import load_data
from model import NeuralNet
from train import train_sgd

def experiment_2b():
    zip_path = '../datasets/bank-note.zip'
    train_file = 'bank-note/train.csv'
    test_file = 'bank-note/test.csv'
    X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

    widths = [5, 10, 25, 50, 100]
    gamma0 = 0.1
    d = 1

    for width in widths:
        print(f"Training width = {width}")
        model = NeuralNet(input_dim=X_train.shape[1], hidden_width=width)
        train_err, test_err, losses = train_sgd(model, X_train, y_train, X_test, y_test, gamma0, d)

        print(f"[Gaussian Init] Train Error: {train_err:.4f}, Test Error: {test_err:.4f}")

        plt.plot(losses, label=f"width={width}")

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss per Epoch (Gaussian weight init)")
    plt.legend()
    plt.grid(True)
    plt.savefig('Q2b.png')
    plt.show()

def experiment_2c():
    zip_path = '../datasets/bank-note.zip'
    train_file = 'bank-note/train.csv'
    test_file = 'bank-note/test.csv'
    X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

    widths = [5, 10, 25, 50, 100]
    gamma0 = 0.1
    d = 1

    for width in widths:
        print(f"Training width = {width}")
        model = NeuralNet(input_dim=X_train.shape[1], hidden_width=width)
        model.set_zero_weights(input_dim=X_train.shape[1])
        train_err, test_err, losses = train_sgd(model, X_train, y_train, X_test, y_test, gamma0, d)

        print(f"[Zero Init] Train Error: {train_err:.4f}, Test Error: {test_err:.4f}")

        plt.plot(losses, label=f"width={width}")

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss per Epoch (Zero weight init)")
    plt.legend()
    plt.grid(True)
    plt.savefig('Q2c.png')
    plt.show()

if __name__ == "__main__":
    experiment_2b()
    experiment_2c()
