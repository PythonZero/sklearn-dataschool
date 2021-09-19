import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# ----- SCORING ----- #
# Method 1 - (Manual) train-test-split & calculate using - % Accuracy / MSE / RMSE / MAE
def measure_accuracy(model, x, y_actual):
    y_predicted = model.predict(x)
    accuracy = metrics.accuracy_score(y_actual, y_predicted)  # % correct
    mae = metrics.mean_absolute_error(y_actual, y_predicted)
    mse = metrics.mean_squared_error(y_actual, y_predicted)
    root_mse = np.sqrt(metrics.mean_squared_error(y_actual, y_predicted))
    print(
        f"{model.__class__.__name__} {accuracy=: .1%} {mae=: .2f} {mse=: .2f} {root_mse=: .3f}"
    )


# Method 2 - (Automatic) Use Cross-Validation w/Stratified sampling
def measure_cross_val_score(model, x, y):
    # cv uses K-Folds to create 10 sets of unique TEST groups. e.g. say you have y = 1 to 50
    #  K1 trains w/[1..45] tests w/[46..50].
    #  K2 trains w/[1..40, 46..50] tests w/[41..45]
    # * Default is stratified sampling - i.e. if 30% of data is cats / 70% dogs, tries to keep that ratio
    cv_accuracy = cross_val_score(model, x, y, cv=10, scoring="accuracy").mean()
    cv_negative_mse = cross_val_score(
        model, x, y, cv=10, scoring="neg_mean_squared_error"
    )
    cv_rmse = np.sqrt(-cv_negative_mse).mean()  # flip the sign & sqrt
    print(f"{model.__class__.__name__} {cv_accuracy=: .1%} {cv_rmse=: .2f}")
