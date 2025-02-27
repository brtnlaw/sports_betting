# Use pickle to store models
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split

MODELS = {
    "linear_regression": LinearRegression(),
    "light_gbm": lgb
}
RANDOM_SEED = 12345

def train_model(X_train, y_train, model_name="linear_regression"):
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from {list(MODELS.keys())}")
    model = MODELS[model_name]
    if model_name in ["linear_regression"]:
        model.fit(X_train, y_train)
    elif model_name in ["light_gbm"]:
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": {"l2", "l1"},
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0,
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        model = model.train( 
            params, lgb_train
        )
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(y_pred)
    print(y_test)
    print(f"Model Performance ({model.__class__.__name__}):")
    print("R2 Score:\n", r2)

# TODO: add in all the features
def train_and_pkl(X, y, model_name, pkl_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = train_model(X_train, y_train, model_name)
    evaluate_model(model, X_test, y_test)
    # with open(f"models/{pkl_name}.pkl", "wb") as f:
    #     pickle.dump(model, f)