# Use pickle to store models
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit

MODELS = {"linear_regression": LinearRegression(), "light_gbm": lgb}
RANDOM_SEED = 12345


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_name="light_gbm"):
    if model_name not in MODELS:
        raise ValueError(
            f"Model '{model_name}' not recognized. Choose from {list(MODELS.keys())}"
        )
    model = MODELS[model_name]
    if model_name in ["linear_regression"]:
        model.fit(X_train, y_train)
    elif model_name in ["light_gbm"]:
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",  # quantile? and see if there's a difference round up, get the quantile
            # so like if it's 3.5 O/U 50%, i plug in 4 to the over and see 55%, then bet there
            "metric": {"l2", "l1"},
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0,
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        model = model.train(params, lgb_train)
    return model


# TODO: set up a baseline and compare, so do something like time based of ---train-----test-future
def evaluate_model_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    if model.__class__.__name__ == "Booster":
        # Get the most predictive features
        feature_importances = model.feature_importance(importance_type="gain")
        feature_names = model.feature_name()
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        importance_df["Percentage"] = (
            importance_df["Importance"] / importance_df["Importance"].sum()
        ) * 100
        top_n_features = importance_df.sort_values(
            by="Importance", ascending=False
        ).head(3)
        print(top_n_features[["Feature", "Percentage"]])
    print("===============================")
    print(f"Model Performance ({model.__class__.__name__}):")
    print("R2 Score:\n", r2)
    print("MSE:\n", mse)
    print("MAE:\n", mae)


def evaluate_betting_metrics():
    # Take model, X_test, y_test --> for each row do the following
    # Get the model pred + probability of the line
    pass


# Perhaps leave 2023, 2024 for after?
def cv(df, col="home_h1"):
    # data
    X = df.drop(columns=[col])
    y = df[col]
    # Step 1: Find indices corresponding to the start of each year.
    cv_year_indices = df.groupby("season").head(1).index
    # Step 2: Perhaps starting with like 2015, train on prior data and test on following year.
    init_train_yrs = 3
    cv_year_indices = [sum(cv_year_indices[:init_train_yrs])] + cv_year_indices[
        init_train_yrs:
    ]
    for idx in cv_year_indices:
        X_train = X.iloc[:idx]
        y_train = y.iloc[:idx]
        X_test = X.iloc[idx:(next)]
        y_test = None
        # Step 3: Develop rudimentary metrics on predicting y_hat_test to y_test
        model = train_model(X_train, y_train)
        evaluate_model_metrics(model, X_test, y_test)
        # TODO: Step 4: Build simple betting logic with the lines instead of just how much we're erroring
        evaluate_betting_metrics(model, X_test, y_test)

        # evaluate_betting_metrics(model, X_test, y_test)
    pass


def train_and_pkl(X, y, model_name="light_gbm", pkl_name="test"):
    # TODO: TimeSeriesSplit this, but in each fold, what bet would be placed?
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    model = train_model(X_train, y_train, model_name)
    evaluate_model_metrics(model, X_test, y_test)
    with open(f"src/cfb/model/models/{pkl_name}.pkl", "wb") as f:
        pickle.dump(model, f)
