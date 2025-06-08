import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 1: Load Data
# -----------------------------
def load_data(filepath):
    dataset = pd.read_csv(filepath)
    X = dataset["area_sqr_ft"].values.reshape(-1, 1)
    y = dataset["price_lakhs"].values.reshape(-1, 1)
    return X, y

# -----------------------------
# Step 2: Define the Model
# -----------------------------
class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.001, n_iters=10000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 100 == 0:
                loss = self._mse(y, y_pred)
                print(f"Iteration {i}: Loss = {loss:.4f}")

                if np.isnan(self.w).any() or np.isnan(self.b):
                    print(f"NaN detected at iteration {i}")
                    break

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


# -----------------------------
# Step 3: Train the Model
# -----------------------------
if __name__ == "__main__":
    X, y = load_data(r"C:\Users\hp\Ml_AI_PRACTICE\ml_model\linear_regression\home_prices.csv")

    # Scaling both features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    model = LinearRegressionFromScratch(learning_rate=0.001, n_iters=10000)
    model.fit(X_scaled, y_scaled)

    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    
    r2_score = model.score(y, y_pred)
    print(f"Model Accuracy (R² Score): {r2_score:.4f}")
    # -----------------------------
    # Step 4: Visualize Results
    # -----------------------------
    plt.scatter(X, y, label="Actual")
    plt.plot(X, y_pred, color="red", label="Prediction")
    plt.xlabel("Area (sq ft)")
    plt.ylabel("Price (lakhs)")
    plt.title("Linear Regression Fit (with Feature Scaling)")
    plt.legend()
    plt.show()

