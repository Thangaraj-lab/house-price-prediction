import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_preprocessing import load_data

def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    }

def run_project():
    df = load_data()

    X = df.drop("medv", axis=1)
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    result_df = pd.DataFrame(results)
    print(result_df)

    import os
    os.makedirs("outputs", exist_ok=True)

    result_df.to_csv("outputs/model_comparison.csv", index=False)

    best_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)

    plt.figure(figsize=(8,5))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.savefig("outputs/prediction_plot.png")
    plt.close()

    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    )

    plt.figure(figsize=(10,6))
    importance.sort_values().plot(kind="barh")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    print("Project Completed Successfully")
