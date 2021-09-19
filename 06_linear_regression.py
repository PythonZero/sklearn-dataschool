import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def print_linear_regression_formula(model):
    print(
        f"y =",
        " + ".join(
            [f"({m:.3f} * {col})" for col, m in zip(feature_cols, linreg.coef_)]
        ),
        f"+ {model.intercept_: .3f}",
    )


def calculate_errors(true, pred):
    mae = metrics.mean_absolute_error(true, pred)
    mse = metrics.mean_squared_error(true, pred)
    root_mse = np.sqrt(metrics.mean_squared_error(true, pred))
    print(f"{mae=: .2f} {mse=: .2f} {root_mse=: .3f}")


if __name__ == "__main__":
    df = pd.read_csv("advertising.csv", index_col=0)
    feature_cols = ["TV", "Radio", "Newspaper"]
    X = df[feature_cols]
    y = df["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Linear Regression: y = m1x1 + m2x2+ ... + c       (y = B1x1 + ... + B0)
    #   e.g.             y = (TV * m1) + (Radio * m2) + (Newspaper * m3) + c
    #   where m1 (or B1) is the model coefficients (which are learnt from least squares)

    linreg = LinearRegression()  # Learn the coefficients
    linreg.fit(X_train, y_train)
    calculate_errors(y_test, linreg.predict(X_test))

    # This tells us that for each $1000 increase in spend TV unit, we get 47 more sales (0.047 * TV)
    print_linear_regression_formula(linreg)

    # Testing different features (no Newspaper) -> smaller error -> Newspaper is not a good indicator
    linreg_2features = linreg.fit(X_train[["TV", "Radio"]], y_train)
    calculate_errors(y_test, linreg_2features.predict(X_test[["TV", "Radio"]]))
