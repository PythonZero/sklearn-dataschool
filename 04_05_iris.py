from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def measure_accuracy(model, x, y_actual, additional_comment=""):
    y_predicted = model.predict(x)
    accuracy = metrics.accuracy_score(y_actual, y_predicted)
    print(f"{model.__class__.__name__} {additional_comment} Accuracy {accuracy: .1%}")


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
    X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]  # to predict

    # Fit K-Nearest neighbors
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X, y)
    print(f"knn prediction: {knn.predict(X_new)}")
    measure_accuracy(knn, X, y,)

    # Fit Logistic Regression
    logreg = LogisticRegression(solver="liblinear")
    logreg.fit(X, y)
    print(f"logreg prediction: {logreg.predict(X_new)}")
    measure_accuracy(logreg, X, y)

    # Use Train/Test split
    knn_train_test = KNeighborsClassifier(n_neighbors=6)
    knn_train_test.fit(X_train, y_train)
    measure_accuracy(knn_train_test, X_test, y_test, "train/test")

    logreg_train_test = LogisticRegression(solver="liblinear")
    logreg_train_test.fit(X_train, y_train)
    measure_accuracy(logreg_train_test, X_test, y_test, "train/test")

    # Ads/Dis of train-test split
    # * Dis: High Variance estimate of out-of-sample accuracy
    # * Dis: Use K-fold cross validation to overcome this
    # * Ads: Fast
