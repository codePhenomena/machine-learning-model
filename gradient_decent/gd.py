import pandas as pd
import numpy as np


def gradient_descent(x,y,lr=0.01,epochs=3000):

    x_min,x_max=x.min(), x.max()
    y_min,y_max=y.min(), y.max()

    x_scaled =(x-x_min)/(x_max-x_min)
    y_scaled=(y-y_min)/(y_max-y_min)

     # Initialize parameters
    b=0.0
    m=0.0
    n=len(y_scaled) 

    for epoch in range(epochs):
        y_pred = b+ m * x_scaled # predication
        error=y_scaled-y_pred  # err
        cost=np.mean(error**2) #mse

        # calculating gradient
        db=-2*np.mean(error) # Derivative w.r.t. intercept b
        dm=-2*np.mean(error *x_scaled) # Derivative w.r.t. slope m
        
        # updating parameters
        b -=lr*db
        m -=lr*dm 

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}, b = {b}, m = {m}")

    b_original = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)
    m_original = m * (y_max - y_min) / (x_max - x_min)

    return b_original, m_original

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\hp\Ml_AI_PRACTICE\ml_model\linear_regression\home_prices.csv")

    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()

    b, m = gradient_descent(x, y)

    print(f"Final Results: m={m}, b={b}")



