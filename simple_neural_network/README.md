# 🧠 Bonus Prediction with PyTorch Neural Network

This project demonstrates a simple **neural network regression model** built using **PyTorch** to predict an employee's **bonus** based on their:

- Performance score
- Years of experience
- Number of projects completed

---

## 📌 Objective

To predict the amount of **bonus** an employee should receive, based on performance metrics, using a feedforward neural network. The goal is to minimize prediction error using **Mean Squared Error (MSE)**.

---

## 🧾 Dataset Structure

The model uses a CSV file named `bonus.csv` with the following features:

| Feature               | Description                                |
|------------------------|--------------------------------------------|
| `performance`          | Numeric performance score of employee      |
| `years_of_experience`  | Number of years the employee has worked    |
| `projects_completed`   | Count of completed projects                |
| `bonus`                | Target variable (monetary bonus in float)  |

---

## 🧠 Model Overview

- **Architecture**: A simple neural network with:
  - 3 input neurons (features)
  - 1 output neuron (bonus)
  - No hidden layers (pure linear regression via neural network)

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Epochs**: 5000 iterations
- **Learning Rate**: 0.005

---

## 🔁 Workflow Summary

1. **Load and explore** the dataset using `pandas`.
2. **Split** the data into training and testing sets using `train_test_split`.
3. **Convert** the data into PyTorch tensors.
4. **Define** the neural network model using `nn.Sequential`.
5. **Train** the model using a manual loop and backpropagation.
6. **Evaluate** the model on the test set and display the test loss.
7. **Inspect** model predictions and parameters.

---

## 📈 Model Output

- During training, the script prints loss every 100 epochs.
- After training, it outputs the **final test loss**.
- It also displays:
  - Sample predictions vs actual bonus values.
  - Learned weights (parameters) of the model.

---

## 🔧 Requirements

- Python 3.x
- PyTorch
- pandas
- scikit-learn
- numpy

Install via:

```bash
pip install torch pandas scikit-learn numpy
