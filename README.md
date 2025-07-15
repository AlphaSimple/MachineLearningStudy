# Machine Learning Study
This is a machine learning library developed by **Aman Sinha** for 
CS5350/6350 in University of Utah.

## Decision Tree Learning Usage

The `DecisionTree` directory contains code for learning decision trees using the ID3 algorithm with support for information gain, majority error, and gini index splitting criteria. You can use the code in two main ways:

### 1. Running Experiment Scripts

- To run experiments on the car evaluation dataset:
  ```sh
  python DecisionTree/Q2.py
  ```
  This will train decision trees with different criteria and depths, and print a summary table of errors.

- For the bank marketing dataset (with numerical and categorical features):
  ```sh
  python DecisionTree/Q3a.py   # For 'unknown' as a value
  python DecisionTree/Q3b.py   # For 'unknown' as missing (imputed)
  ```

### 2. Using the DecisionTree Class Programmatically

You can import and use the `DecisionTree` class in your own scripts:

```python
from DecisionTree.id3 import DecisionTree

tree = DecisionTree(criterion='information_gain', max_depth=3)
# data: list of dicts, attributes: list of feature names
# label_name: the key for the label in each dict (default 'label')
tree.fit(data, attributes, label_name='label')
predictions = tree.predict(data)
```

#### Parameters:
- `criterion`: Splitting criterion. One of `'information_gain'`, `'majority_error'`, `'gini_index'`.
- `max_depth`: Maximum tree depth (int or None for fully grown tree).
- `fit(data, attributes, label_name='label')`: Train the tree. `data` is a list of dictionaries, `attributes` is a list of feature names, and `label_name` is the key for the label in each dictionary.
- `predict(data)`: Predict labels for a list of examples (list of dictionaries).

### Data Loading

Use the provided data loader functions in `DecisionTree/data_loader.py` to load datasets from zip files:

```python
from DecisionTree.data_loader import load_car_data, load_bank_data
car_data = load_car_data('datasets/car-4.zip', split='train')
bank_data = load_bank_data('datasets/bank-4.zip', split='train')
```
- Both loaders return a list of dictionaries, where each dictionary represents an example with attribute names as keys.

See the experiment scripts in the `DecisionTree` directory for more usage examples.

---

## Ensemble Learning Usage

The `Ensemble Learning` directory contains implementations of ensemble algorithms:
- **Bagging (Bagged Trees)**
- **Random Forests**
- **AdaBoost**

### 1. Running Experiment Scripts

- To run bagging, boosting, and random forest experiments on the bank marketing dataset:
  ```sh
  python "Ensemble Learning/Q2a.py"   # AdaBoost
  python "Ensemble Learning/Q2b.py"   # Bagged Trees
  python "Ensemble Learning/Q2c.py"   # Bagging bias-variance analysis
  python "Ensemble Learning/Q2d.py"   # Random Forests
  python "Ensemble Learning/Q2e.py"   # Random Forest bias-variance analysis
  ```
  These scripts will train the respective ensemble models, print results, and save error plots (e.g., `adaboost_ensemble_error.png`).

#### Parameters for Ensemble Scripts
- You can set the number of trees/iterations (T) and feature subset size (k) by editing the script variables (e.g., `T_MAX`, `K_LIST`, or `K` in the script).
- The scripts use the bank marketing dataset and handle all preprocessing internally.

### 2. Using the Ensemble Classes Programmatically

You can import and use the ensemble classes in your own scripts:

```python
from Ensemble Learning.adaboost import AdaBoost
from Ensemble Learning.bagged_trees import BaggedTrees
from Ensemble Learning.random_forest import RandomForest
from Ensemble Learning.decision_stumps import DecisionStump

# AdaBoost
ada = AdaBoost(T=100)
ada.fit(X, y, features, T=100)
preds = ada.predict(X_test)

# Bagged Trees
bagger = BaggedTrees(T=100)
bagger.fit(X, y, features, T=100)
preds = bagger.predict(X_test)

# Random Forest
rf = RandomForest(T=100, k=4)
rf.fit(X, y, features, T=100, k=4)
preds = rf.predict(X_test)

# Decision Stump (weighted, for AdaBoost)
stump = DecisionStump()
stump.fit(data, attributes, label_name='label', sample_weight=weights)
preds = stump.predict(X_test)
```
- `X` is a list of feature dicts, `y` is a list of labels, `features`/`attributes` is a list of feature names.
- For AdaBoost, `data` is a list of dicts with a `'label'` key, and `weights` is a numpy array of sample weights.
- See the experiment scripts for more usage examples and data preprocessing.

---

## Linear Regression Usage (LMS: Batch and Stochastic Gradient Descent)

The `Linear Regression` directory contains code for linear regression using batch and stochastic gradient descent, as well as the analytical (normal equation) solution.

### 1. Running Experiment Scripts

- To run linear regression experiments:
  ```sh
  python "Linear Regression/Q4a.py"   # Batch gradient descent
  python "Linear Regression/Q4b.py"   # Stochastic gradient descent
  python "Linear Regression/Q4c.py"   # Analytical solution (normal equation)
  ```
  These scripts will train linear regression models, print results, and save plots of the cost function history.

#### Parameters for Linear Regression Scripts
- You can set the learning rate (`r`), tolerance (`tol`), and maximum iterations (`max_iter`) by editing the script variables or function arguments.
- The scripts use the provided data loader and handle preprocessing.

### 2. Using the Linear Regression Code Programmatically

You can import and use the gradient descent functions and analytical solution in your own scripts:

```python
from Linear Regression.gradient_descent import batch_gradient_descent, stochastic_gradient_descent, compute_cost

# Batch Gradient Descent
w, b, cost_history, steps = batch_gradient_descent(X, y, r=0.1, tol=1e-6, max_iter=10000)

# Stochastic Gradient Descent
w, b, cost_history, updates = stochastic_gradient_descent(X, y, r=0.1, tol=1e-6, max_iter=10000)

# Analytical Solution (Normal Equation)
def compute_optimal_weights(X, y):
    X_tilde = np.hstack([X, np.ones((X.shape[0], 1))])
    w_tilde = np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ y
    w = w_tilde[:-1]
    b = w_tilde[-1]
    return w, b
w_opt, b_opt = compute_optimal_weights(X, y)
```
- `X` is a numpy array of features, `y` is a numpy array of targets.
- `w` is the weight vector, `b` is the bias term.
- See the experiment scripts for more usage examples and data preprocessing.

---

## Perceptron Usage

The `Perceptron` directory contains implementations of the perceptron learning algorithms, including standard, voted, and averaged perceptron.

### 1. Running Experiment Scripts

- To run perceptron experiments:
  ```sh
  python Perceptron/Q2.py
  ```
  This script will train perceptron models and print results such as accuracy or error rates.

#### Parameters for Perceptron Scripts
- You can set the number of epochs (`T`) and learning rate (`r`) by editing the script variables or function arguments.
- The script uses the provided data loader and handles preprocessing.

### 2. Using the Perceptron Algorithms Programmatically

You can import and use the perceptron algorithms in your own scripts:

```python
from Perceptron.perceptron_algorithms import standard_perceptron, voted_perceptron, averaged_perceptron

# Standard Perceptron
w = standard_perceptron(X_train, y_train, T=10, r=1.0)

# Voted Perceptron
weights = voted_perceptron(X_train, y_train, T=10, r=1.0)
# weights is a list of (weight vector, count) tuples

# Averaged Perceptron
avg_w = averaged_perceptron(X_train, y_train, T=10, r=1.0)
```
- `X_train` is a numpy array of features, `y_train` is a numpy array of labels.
- See the experiment script for more usage examples and data preprocessing.

---

## SVM Usage

The `SVM` directory contains implementations of Support Vector Machine algorithms, including:
- Primal SVM (SGD-based)
- Dual SVM (Quadratic Programming)
- Kernel Perceptron (for nonlinear SVM)

### 1. Running Experiment Scripts

- To run SVM experiments:
  ```sh
  python SVM/Q2.py      # Primal SVM (SGD)
  python SVM/Q3a.py     # Dual SVM (QP)
  python SVM/Q3b.py     # Kernel Perceptron
  python SVM/Q3c.py     # Additional SVM experiments
  python SVM/Q3d.py     # More SVM experiments
  ```
  These scripts will train SVM models and print results such as accuracy, error rates, or save plots.

#### Parameters for SVM Scripts
- You can set the learning rate, number of epochs, regularization parameter (C), kernel type, and other hyperparameters by editing the script variables (e.g., `gamma_0`, `a`, `T`, `C`, `gamma`, `schedule_type`).
- The scripts use the provided data loader and handle preprocessing.

### 2. Using the SVM Algorithms Programmatically

You can import and use the SVM algorithms in your own scripts:

```python
from SVM.svm_sgd import svm_sgd
from SVM.svm_dual import svm_dual, svm_dual_gaussian, predict_gaussian
from SVM.kernel_perceptron import kernel_perceptron, predict_kernel_perceptron

# Primal SVM (SGD)
w, obj_values = svm_sgd(X_train, y_train, C=1.0, gamma_0=0.01, a=1.0, T=100, schedule_type='part-a')

# Dual SVM (QP, linear kernel)
alpha, w, b = svm_dual(X_train, y_train, C=1.0)

# Dual SVM (Gaussian kernel)
alpha, b = svm_dual_gaussian(X_train, y_train, C=1.0, gamma=0.5)
y_pred = predict_gaussian(X_test, X_train, y_train, alpha, b, gamma=0.5)

# Kernel Perceptron (Gaussian kernel)
c = kernel_perceptron(X_train, y_train, gamma=0.5, T=10)
y_pred = predict_kernel_perceptron(X_test, X_train, y_train, c, gamma=0.5)
```
- `X_train`, `X_test` are numpy arrays of features, `y_train` is a numpy array of labels.
- See the experiment scripts for more usage examples and data preprocessing.

---

## Neural Networks Usage

The `Neural Networks` directory contains code for training and evaluating a 3-layer neural network (two hidden layers, one output) using stochastic gradient descent (SGD).

### 1. Running Experiment Scripts

- To run neural network experiments:
  ```sh
  python Neural\ Networks/train.py
  ```
  (Or run the appropriate experiment script if provided.)
  This will train the neural network and print or save results such as training/test error and loss curves.

#### Parameters for Neural Network Scripts
- You can set the learning rate (`gamma0`), learning rate schedule parameter (`d`), number of epochs (`max_epochs`), and hidden layer width by editing the script variables or function arguments.
- The scripts use numpy arrays for data and handle preprocessing.

### 2. Using the Neural Network Code Programmatically

You can import and use the neural network model and training function in your own scripts:

```python
from Neural Networks.model import NeuralNet
from Neural Networks.train import train_sgd

# Example usage:
model = NeuralNet(input_dim=X_train.shape[1], hidden_width=10)
train_error, test_error, train_losses = train_sgd(
    model, X_train, y_train, X_test, y_test,
    gamma0=0.01, d=1.0, max_epochs=20
)
```
- `X_train`, `X_test` are numpy arrays of features, `y_train`, `y_test` are numpy arrays of labels.
- See the experiment scripts for more usage examples and data preprocessing.

---

## Logistic Regression Usage

The `Logistic Regression` directory contains code for logistic regression using stochastic gradient descent (SGD) for both maximum likelihood (ML) and maximum a posteriori (MAP) estimation.

### 1. Running Experiment Scripts

- To run logistic regression experiments:
  ```sh
  python "Logistic Regression/Q3a.py"   # MAP estimation (with prior)
  python "Logistic Regression/Q3b.py"   # ML estimation (no prior)
  ```
  These scripts will train logistic regression models, print train/test errors, and save plots of the objective/loss.

#### Parameters for Logistic Regression Scripts
- You can set the learning rate (`gamma_0`), learning rate schedule parameter (`d`), number of epochs (`T`), and prior variance (`v` for MAP) by editing the script variables or function arguments.
- The scripts use the provided data loader and handle preprocessing, including adding a bias term.

### 2. Using the Logistic Regression Code Programmatically

You can import and use the training and evaluation functions in your own scripts:

```python
from Logistic Regression.Q3a import train_map_sgd, evaluate
from Logistic Regression.Q3b import train_ml_sgd, evaluate

# MAP Estimation (with prior)
w, obj_vals = train_map_sgd(X_train, y_train, v=0.1, gamma_0=0.01, d=1, T=100)
train_err = evaluate(X_train, y_train, w)
test_err = evaluate(X_test, y_test, w)

# ML Estimation (no prior)
w, obj_vals = train_ml_sgd(X_train, y_train, gamma_0=0.01, d=1, T=100)
train_err = evaluate(X_train, y_train, w)
test_err = evaluate(X_test, y_test, w)
```
- `X_train`, `X_test` are numpy arrays of features (with bias term added), `y_train`, `y_test` are numpy arrays of labels.
- See the experiment scripts for more usage examples and data preprocessing.
