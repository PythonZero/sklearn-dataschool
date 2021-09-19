import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import binarize


def initialise_data(feature_columns):
    X = df[feature_columns]
    y = df.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X, y, X_train, X_test, y_train, y_test


def fit_logistic_regression(X_train, y_train):
    logistic_regression_model = LogisticRegression(solver="liblinear")
    logistic_regression_model.fit(X_train, y_train)
    return logistic_regression_model


def measure_accuracy(model, input_x, y_actual):
    y_predictions = model.predict(input_x)
    accuracy = sklearn.metrics.accuracy_score(y_actual, y_predictions)
    print(f"Accuracy of {model.__class__.__name__}: {accuracy: .1%}")
    measure_null_accuracy(y_actual)


def measure_null_accuracy(y):
    """The accuracy if you just assume everything is the most frequent class"""
    # print(y.value_counts())
    most_frequent_class_count = y.value_counts().max()
    all_count = y.count()
    null_accuracy = most_frequent_class_count / all_count
    print(f"Null Accuracy: {null_accuracy: .1%}")


def print_confusion_matrix(y_actual, y_predicted):
    """SKLearn defaults to labels in alphabetical order (0 then 1).
    You can specify the labels if you want.

                  Predicted 0      Predicted 1
        Actual 0 [118 (TN)              12 (FP) ]
        Actual 1 [47  (FN)              15 (TP) ]

    True Positive (TP) - correctly predicted they have diabetes
    True Negative (TN) - correctly predicted they don't have diabetes
    False Positive (FP) - incorrectly predicted they have diabetes
    False Negative (FN) - incorrectly predicted they don't have diabetes

    """
    confusion_matrix = sklearn.metrics.confusion_matrix(y_actual, y_predicted)
    print(f"Confusion Matrix".center(30, "="), confusion_matrix, sep="\n")
    TN, FP, FN, TP = confusion_matrix.ravel()
    print(f"Classification Accuracy: {(TN + TP )/(TN + TP + FN + FP): .0%}")
    # Sensitivity: When actually positive, how often is it correct
    print(f"Sensitivity: {TP / (TP + FN) : .0%}")
    # Specificity: When actually negative, how often correct?
    print(f"Specificity: {TN / (FP + TN): .0%}")
    # False Positive: (opposite of Specificity) - When negative, how often wrong
    print(f"False Positive: {FP / (TN + FP): .0%}")
    # Precision: When predicting Positive, how often is it correct?
    print(f"Precision: {TP/ (FP +TP): .0%}")
    print("=" * 30)

    # Metric to focus on depends on Biz objective
    # * Spam Filter - either Sensitivity & Precision (as its better for FN than FP - i.e. dont go junk)
    # * Fraud Detector - Sensitivity - FP better than FN (better to flag as fraud vs go undetected)


def plot_probabilities(y_pred_probs):
    plt.hist(y_pred_probs, bins=8)
    plt.xlim(0, 1)
    plt.title("Histogram of predicted probabilities")
    plt.xlabel("Predicted probability of diabetes")
    plt.ylabel("Frequency")
    plt.show()


def plot_roc(y_actual, y_predicted):
    """ROC plots can help you choose the threshold to balance Sensitivity & Specificity.
    Can't see the actual thresholds on the curve itself"""
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_actual, y_predicted)
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()
    return false_positive_rate, true_positive_rate, thresholds


def evaluate_threshold(thresholds, threshold: float):
    print(f"{threshold=}")
    print(f"Sensitivity: {true_positive_rate[thresholds > threshold][-1]: .0%}")
    print(f"Specificity: {1 - false_positive_rate[thresholds > threshold][-1]: .0%}")
    print()


if __name__ == "__main__":
    df = pd.read_csv("pima-indian-diabetes.csv")

    # Can we predict diabetes based off health variables
    HEALTH_FEATURE_COLS = ["pregnant", "insulin", "bmi", "age"]
    X, y, X_train, X_test, y_train, y_test = initialise_data(HEALTH_FEATURE_COLS)
    logistic_regression_model = fit_logistic_regression(X_train, y_train)

    # Compare accuracies (model vs null accuracy)
    measure_accuracy(logistic_regression_model, X_test, y_test)

    # Create confusion matrix
    y_predictions = logistic_regression_model.predict(X_test)
    print_confusion_matrix(y_test, y_predictions)

    # Get the probabilities that each is 0 or 1 (the probabilties are inverse, e.g. 0.21 vs 0.79)
    y_prediction_probabilities = logistic_regression_model.predict_proba(X_test)
    y_probabilty_of_trues = y_prediction_probabilities[:, 1]  # get the Trues (1)
    plot_probabilities(y_probabilty_of_trues)

    # Change prediction threshold to 0.3 (default is 0.5)
    y_true_if_30_pct_confident = binarize([y_probabilty_of_trues], threshold=0.3)[0]
    print("\n\nClassification Accuracy at 30% sensitivity ")
    print_confusion_matrix(y_test, y_true_if_30_pct_confident)

    # ROC and AUC curves (Receiver Operating Characteristic & Area Under curve)
    false_positive_rate, true_positive_rate, thresholds = plot_roc(y_test, y_probabilty_of_trues)
    evaluate_threshold(thresholds, 0.5)
    evaluate_threshold(thresholds, 0.3)
    print(f"AUC: {metrics.roc_auc_score(y_test, y_probabilty_of_trues):.0%}")
    # * AUC is useful as a single number summary of performance
    # * If you randomly pick 1 +ve & 1 -ve observation, AUC = likelihood classifier will assign
    #   higher predicted probability to the positive observation
    # * AUC useful even when there is high class imbalance (unlike classification accuracy - for spam)

    # Calculate Cross Validated AUC
    cross_val = cross_val_score(logistic_regression_model, X, y, cv=100, scoring="roc_auc").mean()
    print(f"Cross Validation AUC Score: {cross_val: .0%}")

    # Confusion Matrix Advantages:
    # * Allows Calculation wide variety of metrics
    # * Useful for multi-class problems

    # ROC/AUC Advantages:
    # * Does NOT require setting a classification threshold
    # * Still useful when high class imbalance
