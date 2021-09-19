import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def measure_accuracy(model, x, y_actual, additional_comment=""):
    y_predicted = model.predict(x)
    accuracy = metrics.accuracy_score(y_actual, y_predicted)
    print(f"{model.__class__.__name__} {additional_comment} Accuracy {accuracy: .1%}")


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=4
    )

    # K-Folds creates 10 sets of unique TESTS. e.g. say you have y = 1 to 50
    #  K1 trains with 1->45, and tests with 46->50
    #  K2 trains with 1->40, 46->50 and tests with 41->45
    k_folds = KFold(n_splits=5, shuffle=False).split(range(25))

    # Cross_validation_score - (automatically) find the accuracy score across all 10 k-Folds
    # * Uses stratified sampling - i.e. if 30% of data is cats / 70% dogs, tries to keep that ratio
    knn = KNeighborsClassifier(n_neighbors=5)
    cv_accuracy = cross_val_score(knn, X, y, cv=10, scoring="accuracy").mean()

    cv_negative_mse = cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_negative_mse).mean()  # flip the sign & sqrt
