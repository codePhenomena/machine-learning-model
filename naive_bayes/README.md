# 📧 Naive Bayes Spam Classifier (From Scratch)

This project demonstrates how to build a simple **Naive Bayes classifier from scratch in Python** to detect spam emails using basic natural language processing (NLP) techniques.

---

## 📌 Objective

Build a machine learning model that classifies emails as **"spam"** or **"ham"** (not spam) using word frequencies learned from labeled training data.

---

## 📊 Dataset Structure

The classifier expects a CSV file named `spam.csv` with the following columns:

| text                                | label |
|-------------------------------------|-------|
| "Win a free iPhone now"             | spam  |
| "Please find attached the report"   | ham   |

- `text`: The email body.
- `label`: `"spam"` or `"ham"`.

---

## ⚙️ How It Works

1. **Preprocessing**:
   - Convert text to lowercase.
   - Remove special characters and digits.
   - Tokenize text into words.

2. **Bag of Words Model**:
   - Vocabulary is built from training emails.
   - Each email is vectorized using word counts.

3. **Training (Naive Bayes)**:
   - Calculates word probabilities per class (`P(word|spam)`, `P(word|ham)`).
   - Computes class priors (`P(spam)`, `P(ham)`).
   - Applies **Laplace smoothing** to avoid zero probabilities.

4. **Prediction**:
   - Uses **log probabilities** to avoid underflow.
   - Chooses the label with the higher log-likelihood.

---

## 🚀 How to Run

```bash
# Step 1: Install dependencies (if needed)
pip install pandas numpy scikit-learn

# Step 2: Run the Python script
python naive_bayes_spam_classifier.py
