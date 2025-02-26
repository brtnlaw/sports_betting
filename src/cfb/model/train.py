# Use pickle to store models
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

MODELS = {
    "linear_regression": LinearRegression()
}
RANDOM_SEED = 12345

def train_model(X_train, y_train, model_name="linear_regression"):
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from {list(MODELS.keys())}")
    model = MODELS[model_name]
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance ({model.__class__.__name__}):")
    print("R2 Score:\n", r2)

# TODO: add in all the features
def train_and_pkl(X, y, model_name, pkl_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = train_model(X_train, y_train, model_name)
    evaluate_model(model, X_test, y_test)
    # with open(f"models/{pkl_name}.pkl", "wb") as f:
    #     pickle.dump(model, f)