import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV


def plot_knn_accuracy():
    # Plot the effect of K on KNN accuracy
    plt.plot(range(1, 31), grid_results["mean_test_score"])
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Cross-Validated Accuracy")
    plt.show()


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    # Parameters that KNN can take (n_neighbors, and weights)
    param_grid = {"n_neighbors": range(1, 31), "weights": ["uniform", "distance"]}
    knn = KNeighborsClassifier(n_neighbors=5)
    grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
    grid.fit(X, y)
    grid_results = pd.DataFrame(grid.cv_results_)[
        ["mean_test_score", "std_test_score", "params"]
    ]
    print(f"{grid.best_score_=: .2f} {grid.best_params_=}, {grid.best_estimator_}")

    # plot_knn_accuracy()

    # Less Computationally expensive (Use RandomizedSearchCV)
    rand_grid = RandomizedSearchCV(
        knn, param_grid, cv=10, scoring="accuracy", n_iter=10, random_state=5
    )
    rand_grid.fit(X, y)
    rand_grid_results = pd.DataFrame(rand_grid.cv_results_)[
        ["mean_test_score", "std_test_score", "params"]
    ]
    print(
        f"{rand_grid.best_score_=: .2f} {rand_grid.best_params_=}, {rand_grid.best_estimator_}"
    )

    # Finding best Features (method 1)
    kbest4 = SelectKBest(chi2, k=4).fit_transform(X, y)
    kbest3 = SelectKBest(chi2, k=3).fit_transform(X, y)
    kbest2 = SelectKBest(chi2, k=2).fit_transform(X, y)
    kbest1 = SelectKBest(chi2, k=1).fit_transform(X, y)

    # Find feature importance (scores each feature) -> can use top N features
    model = ExtraTreesClassifier(n_estimators=100, random_state=1)
    model.fit(X, y)
    print(model.feature_importances_)

    # Test all feature combinations
    all_feature_combos = [
        combo
        for i in range(1, len(feature_cols) + 1)
        for combo in itertools.combinations(feature_cols, i)
    ]
