"""
Machine Learning Regression Model for Synchronous Machine
--------------------------------------------------------
Author: Lucas Portillo

This script trains regression models to approximate the steady-state behavior
of a synchronous machine using synthetic data generated from physical equations.

Objective:
- Predict active power (P) and reactive power (Q)
- Demonstrate the use of data-driven surrogate models in power systems

This module represents the first integration of electrical machines modeling
with machine learning techniques.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------
# Load dataset
# -------------------------------------------------

def load_dataset(path="synchronous_machine_dataset.csv"):
    return pd.read_csv(path)


# -------------------------------------------------
# Prepare features and targets
# -------------------------------------------------

def prepare_data(df):
    X = df[["V_terminal_pu", "E_internal_pu", "delta_deg"]]
    y_P = df["P_pu"]
    y_Q = df["Q_pu"]
    return X, y_P, y_Q


# -------------------------------------------------
# Train and evaluate model
# -------------------------------------------------

def train_and_evaluate(X, y, model_name="Linear"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_name == "Linear":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=1.0)
    else:
        raise ValueError("Unsupported model")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2


# -------------------------------------------------
# Main execution
# -------------------------------------------------

if __name__ == "__main__":
    df = load_dataset()
    X, y_P, y_Q = prepare_data(df)

    print("Regression results for Active Power (P):")
    for model_name in ["Linear", "Ridge"]:
        mae, rmse, r2 = train_and_evaluate(X, y_P, model_name)
        print(f"{model_name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    print("\nRegression results for Reactive Power (Q):")
    for model_name in ["Linear", "Ridge"]:
        mae, rmse, r2 = train_and_evaluate(X, y_Q, model_name)
        print(f"{model_name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
