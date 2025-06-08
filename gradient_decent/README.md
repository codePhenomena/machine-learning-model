## 📊 Gradient Descent for Linear Regression (from Scratch)

This project implements **Linear Regression** using **Gradient Descent** from scratch in Python, applying **Min-Max Scaling** for data normalization. It's a great educational example for understanding how optimization works in machine learning without using libraries like `sklearn`.

---

### 📁 Dataset

The script expects a CSV file named:
`home_prices.csv` with at least the following columns:

* `area_sqr_ft`: Area of the house in square feet.
* `price_lakhs`: Price of the house in lakhs.

Example structure:

| area\_sqr\_ft | price\_lakhs |
| ------------- | ------------ |
| 1000          | 50           |
| 1500          | 70           |
| ...           | ...          |

---

### 🧮 How It Works

1. **Min-Max Scaling** is applied to bring `x` and `y` values into the \[0,1] range.
2. **Gradient Descent** updates slope (`m`) and intercept (`b`) over multiple iterations (epochs).
3. After convergence, the model parameters are rescaled back to the original data scale.
4. Every 100 epochs, it prints:

   * Current cost (Mean Squared Error)
   * Current values of `m` and `b`

---

### 📦 Dependencies

* Python 3.x
* `pandas`
* `numpy`

Install them via:

```bash
pip install pandas numpy
```

---

### 🚀 Running the Script

```bash
python gradient_descent_from_scratch.py
```

Ensure you update the path to your dataset if needed:

```python
df = pd.read_csv(r"C:\Users\hp\Ml_AI_PRACTICE\ml_model\linear_regression\home_prices.csv")
```

---

### 📌 Output

You’ll see output like:

```
Epoch 0: Cost = 0.1234, b = 0.04, m = 0.11
Epoch 100: Cost = 0.0123, b = 0.12, m = 0.88
...
Final Results: m = 0.0568, b = 10.23
```

These final `m` and `b` values are scaled back to reflect the original data.

---

### 📚 Concepts Covered

* Gradient Descent
* Mean Squared Error
* Parameter Update Rule
* Min-Max Normalization
* Rescaling parameters back to original scale


