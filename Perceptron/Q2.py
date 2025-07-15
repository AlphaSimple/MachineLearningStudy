from data_loader import load_data
from perceptron_algorithms import standard_perceptron, voted_perceptron, averaged_perceptron
import numpy as np

def main():
    # Load the data
    zip_path = '../datasets/bank-note.zip'
    train_file = 'bank-note/train.csv'
    test_file = 'bank-note/test.csv'
    X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

    T = 10  # Maximum number of epochs

    # Standard Perceptron
    w_standard = standard_perceptron(X_train, y_train, T)
    y_pred_standard = np.sign(np.dot(X_test, w_standard))
    error_standard = np.mean(y_pred_standard != y_test)
    print("Standard Perceptron:")
    print("Learned weight vector:", w_standard)
    print("Average prediction error:", error_standard)
    print()

    # Voted Perceptron
    voted_weights = voted_perceptron(X_train, y_train, T)
    y_pred_voted = []
    for x in X_test:
        prediction = np.sign(sum(c * np.sign(np.dot(w, x)) for w, c in voted_weights))
        y_pred_voted.append(prediction)
    error_voted = np.mean(np.array(y_pred_voted) != y_test)
    print("Voted Perceptron:")
    print("Weight vectors and counts:", voted_weights)
    print("Average prediction error:", error_voted)
    print()

    # Averaged Perceptron
    w_averaged = averaged_perceptron(X_train, y_train, T)
    y_pred_averaged = np.sign(np.dot(X_test, w_averaged))
    error_averaged = np.mean(y_pred_averaged != y_test)
    print("Averaged Perceptron:")
    print("Learned weight vector:", w_averaged)
    print("Average prediction error:", error_averaged)
    print()

    # Comparison
    print("Comparison of average prediction errors:")
    print("Standard Perceptron error:", error_standard)
    print("Voted Perceptron error:", error_voted)
    print("Averaged Perceptron error:", error_averaged)

if __name__ == "__main__":
    main()