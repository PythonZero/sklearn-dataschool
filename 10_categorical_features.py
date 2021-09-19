import itertools

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")
    df = df.loc[df["Embarked"].notna()]

    X = df.drop(["Survived", "Name", "Ticket", "Cabin", "PassengerId"], axis="columns")
    y = df.Survived

    logreg = LogisticRegression(solver="lbfgs")

    # Predicting on a single feature (Passenger class)
    cv_score = cross_val_score(
        logreg, X.loc[:, ["Pclass"]], y, cv=5, scoring="accuracy"
    ).mean()
    null_accuracy = y.value_counts(normalize=True)
    print(f"CV Score: {cv_score: .0%}, Null accuracy: {null_accuracy.max(): .0%}")

    # Encode (Single) Categorical Features
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit_transform(X.loc[:, ["Sex"]])  # e.g. encode Sex
    print(one_hot_encoder.categories_)  # female, male
    one_hot_encoder.fit_transform(X.loc[:, ["Embarked"]])  # overwrite Sex w/Embarked
    print(one_hot_encoder.categories_)  # C, Q, S

    # Impute Nulls
    X.loc[X["Age"].isnull(), "Age"] = np.ceil(X["Age"].mean())

    # Cross Validate a pipeline with all features
    column_transformer = make_column_transformer(
        (OneHotEncoder(), ["Sex", "Embarked"]), remainder="passthrough"
    )
    column_transformer.fit_transform(X)
    logistic_regression = LogisticRegression(solver="lbfgs", max_iter=10000)
    pipeline = make_pipeline(column_transformer, logistic_regression)
    cv_score2 = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy").mean()
    print(f"CV2 Score: {cv_score2: .0%}, Null accuracy: {null_accuracy.max(): .0%}")

    # Predict
    X_new = X.sample(5, random_state=99)
    pipeline.fit(X, y)
    pipeline.predict(X_new)

    # Finding best features
    feature_cols = X.columns
    all_feature_combos = [
        list(combo)
        for i in range(1, len(feature_cols) + 1)
        for combo in itertools.combinations(feature_cols, i)
    ]

    feature_scores = {}
    for feature_combo in all_feature_combos:
        X_test = X.loc[:, feature_combo]
        print(f"Testing: {feature_combo}")
        column_transformer = make_column_transformer(
            (
                OneHotEncoder(),
                [col for col in ["Sex", "Embarked"] if col in feature_combo],
            ),
            remainder="passthrough",
        )
        column_transformer.fit_transform(X_test)
        logistic_regression = LogisticRegression(solver="lbfgs", max_iter=10000)
        pipeline = make_pipeline(column_transformer, logistic_regression)
        cv_score2 = cross_val_score(
            pipeline, X_test, y, cv=5, scoring="accuracy"
        ).mean()
        print(f"CV2 Score: {cv_score2: .0%}, Null accuracy: {null_accuracy.max(): .0%}")

        feature_scores[", ".join(feature_combo)] = cv_score2

    print(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))


    # Finding best features with GridSearchCV