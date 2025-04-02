import datetime as dt
import os
import pickle as pkl
import warnings
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import strategy.betting_logic as betting_logic
from data.data_prep import DataPrep
from pipeline import get_features_and_model_pipeline
from preprocessing import preprocess_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

warnings.simplefilter(action="ignore", category=FutureWarning)


class RollingTimeSeriesSplit(BaseCrossValidator):
    """
    Custom rolling cross-validator that expands the training set season by season.
    """

    def __init__(self, seasons, init_train_yrs=5):
        self.seasons = seasons
        self.init_train_yrs = init_train_yrs
        self.unique_seasons = sorted(seasons.unique())

    def split(self, X, y=None, groups=None):
        """Yields indices for train-test splits."""
        # TODO: Expand this to week by week instead of season by season. Am interested in how it learns throughout the season
        for i in range(self.init_train_yrs, len(self.unique_seasons)):
            train_seasons = self.unique_seasons[:i]
            test_season = self.unique_seasons[i]

            train_idx = np.where(self.seasons.isin(train_seasons))[0]
            test_idx = np.where(self.seasons == test_season)[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splits."""
        return len(self.unique_seasons) - self.init_train_yrs


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    odds_df: pd.DataFrame,
    betting_fnc: Callable = betting_logic.simple_percentage,
    init_train_yrs: int = 5,
    file_name: str = None,
):
    cv_split = RollingTimeSeriesSplit(
        seasons=X["season"], init_train_yrs=init_train_yrs
    )
    for train_idx, test_idx in cv_split.split(X, y):
        # Train-test split
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        pipeline.fit(X_train, y_train)

        # Get predictions
        preds = pipeline.predict(X_test)
        odds_df.iloc[test_idx, odds_df.columns.get_loc("pred")] = preds

    odds_df = betting_fnc(odds_df)

    # Save the entire pipeline
    if not file_name:
        td = dt.datetime.today()
        file_name = f"model_{td.month}_{td.day}_{td.year%100}"
    joblib.dump(pipeline, file_name)

    # Save the odds_df with the betting fnc name
    with open(
        f"src/cfb/model/models/{file_name}_{betting_fnc.__name__}.pkl", "wb"
    ) as f:
        pkl.dump(odds_df, f)
    return pipeline, odds_df


def load_pkl_if_exists(
    name_str, betting_fnc=betting_logic.simple_percentage, file_type="df"
):
    """Helper function to load 'pipeline' or 'df' from a str."""
    assert file_type in ["pipeline", "df"], "Pick a file_type in 'pipeline', 'df'"
    if file_type == "pipeline":
        file_path = f"src/cfb/model/models/{name_str}.pkl"
    else:
        file_path = f"src/cfb/model/models/{name_str}_{betting_fnc.__name__}.pkl"
    if not os.path.exists(file_path):
        raise Exception(f"No properly configured {file_type} file.")
    with open(file_path, "rb") as file:
        result = pkl.load(file)
    return result


def plot_pnl(model_str, betting_fnc=betting_logic.simple_percentage):
    model_df = load_pkl_if_exists(model_str, betting_fnc, "df")
    model_df.fillna(0, inplace=True)
    plot_model_df = model_df[model_df["unit_pnl"] != 0]
    plot_model_df.reset_index(drop=True, inplace=True)

    plt.plot(plot_model_df["unit_pnl"].cumsum(), label=model_str)
    plt.xlabel("Games")
    plt.ylabel("Profit/Loss (units)")
    plt.title("Model Betting Strategy Performance")
    plt.legend()
    plt.show()


def plot_pnl_comparison(
    model_str, baseline_str="model_3_29_25", betting_fnc=betting_logic.simple_percentage
):
    # TODO: Include target column in model name.
    plot_model_df = load_pkl_if_exists(model_str, betting_fnc, "df")
    plot_baseline_df = load_pkl_if_exists(baseline_str, betting_fnc, "df")

    plot_model_df.fillna(0, inplace=True)
    plot_model_df = plot_model_df[plot_model_df["unit_pnl"] != 0]
    plot_model_df.reset_index(drop=True, inplace=True)
    plot_baseline_df.fillna(0, inplace=True)
    plot_baseline_df = plot_baseline_df[plot_baseline_df["unit_pnl"] != 0]
    plot_baseline_df.reset_index(drop=True, inplace=True)

    plt.plot(
        plot_model_df["unit_pnl"].cumsum(), label=f"{model_str}_{betting_fnc.__name__}"
    )
    plt.plot(
        plot_baseline_df["unit_pnl"].cumsum(),
        label=f"{baseline_str}_{betting_fnc.__name__}",
    )
    plt.xlabel("Games")
    plt.ylabel("Profit/Loss (units)")
    plt.title("Betting Strategy Performance")
    plt.legend()
    plt.show()


# TODO: Below needs to be updated.
def model_metrics(
    model_str,
    baseline_str="baseline_3_30_25",
    betting_fnc=betting_logic.simple_percentage,
):
    # NOTE: Max drawdown, Brier score
    plot_pnl_comparison(model_str, baseline_str)

    model = load_pkl_if_exists(model_str, betting_fnc, "model")
    baseline = load_pkl_if_exists(baseline_str, betting_fnc, "model")
    model_df = load_pkl_if_exists(model_str, betting_fnc, "df")
    baseline_df = load_pkl_if_exists(baseline_str, betting_fnc, "df")

    str_list = [model_str, baseline_str]
    df_list = [model_df, baseline_df]
    model_list = [model, baseline]

    print("==============================================")
    print(
        f"PNL delta from the model: {(model_df["unit_pnl"].sum() - baseline_df["unit_pnl"].sum()).round(2)} units"
    )

    for i in range(2):
        df_list[i].reset_index(drop=True, inplace=True)
        df_list[i].dropna(inplace=True)
        print("==============================================")
        print(f"Model Performance ({model.__class__.__name__}): {str_list[i]}")
        y_pred = df_list[i]["pred"]
        y_test = df_list[i]["total"]
        r2 = r2_score(y_test, y_pred)
        print("R2 Score:\n", r2)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE:\n", mse)
        mae = mean_absolute_error(y_test, y_pred)
        print("MAE:\n", mae)
        if model_list[i].__class__.__name__ == "Booster":
            # Get the most predictive features
            feature_importances = model_list[i].feature_importance(
                importance_type="gain"
            )
            feature_names = model_list[i].feature_name()
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


if __name__ == "__main__":
    # python src/cfb/backtest.py
    print("Step 1: Loading data...")
    data_prep = DataPrep(dataset="cfb")
    raw_data = data_prep.get_data()

    print("Step 2: Preprocess and separate odds, X, and y...")
    preprocessed_data = preprocess_pipeline().fit_transform(raw_data)
    target_col = "total"
    betting_cols = ["min_ou", "max_ou"]

    odds_df = preprocessed_data[[target_col] + betting_cols]
    odds_df["pred"] = None
    X = preprocessed_data.drop(columns=[target_col])
    y = preprocessed_data[target_col]

    print("Step 3: Training and evaluating the model...")
    pipeline = get_features_and_model_pipeline()
    model, odds_df = cross_validate(
        X, y, pipeline, odds_df, betting_logic.simple_percentage
    )
