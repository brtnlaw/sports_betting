import datetime as dt
import os
import warnings
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import strategy.betting_logic as betting_logic
from data.data_prep import DataPrep
from pipelines.pipeline import get_features_and_model_pipeline
from pipelines.preprocessing import preprocess_pipeline
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())


class RollingTimeSeriesSplit(BaseCrossValidator):
    """Custom rolling cross-validator that rolls over the training window by season, to avoid model drift."""

    def __init__(self, seasons, fixed_window_length=5):
        self.seasons = seasons
        self.fixed_window_length = fixed_window_length
        self.unique_seasons = sorted(seasons.unique())

    def split(self, X, y=None, groups=None):
        """Yields indices for train-test splits."""
        for i in range(self.fixed_window_length, len(self.unique_seasons)):
            train_seasons = self.unique_seasons[i - self.fixed_window_length : i]
            test_season = self.unique_seasons[i]

            train_idx = np.where(self.seasons.isin(train_seasons))[0]
            test_idx = np.where(self.seasons == test_season)[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splits."""
        return len(self.unique_seasons) - self.fixed_window_length


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    odds_df: pd.DataFrame,
    betting_fnc: Callable = betting_logic.simple_percentage,
    fixed_window_length: int = 5,
    file_name: str = None,
) -> tuple[Pipeline, pd.DataFrame]:
    """
    Cross validates according to splits selected by RollingTimeSeriesSplit above. More specifically,
    fits based on a given block, predicts the next block, and then rolls that block into its fitting
    data. Then outputs both the final pipeline and the df of bets into the models/ folder.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        pipeline (Pipeline): Feature and model pipeline.
        odds_df (pd.DataFrame): DataFrame of betting lines and results.
        betting_fnc (Callable, optional): Function to allocate bets. Defaults to betting_logic.simple_percentage.
        fixed_window_length (int, optional): Seasons to train on. Defaults to 5.
        file_name (str, optional): Desired file name for model. Defaults to None.

    Returns:
        tuple[Pipeline, pd.DataFrame]: The total pipeline that has been fit and the bets made.
    """
    cv_split = RollingTimeSeriesSplit(
        seasons=X["season"], fixed_window_length=fixed_window_length
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

    # Save the pipeline and odds_df with the betting fnc name
    joblib.dump(
        pipeline,
        os.path.join(PROJECT_ROOT, f"src/cfb/models/{file_name}_{y.name}_pipeline.pkl"),
    )
    joblib.dump(
        odds_df,
        os.path.join(
            PROJECT_ROOT,
            f"src/cfb/models/{file_name}_{y.name}_{betting_fnc.__name__}.pkl",
        ),
    )
    return pipeline, odds_df


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
    odds_df.loc[:, "pred"] = None
    X = preprocessed_data.drop(columns=[target_col] + betting_cols)
    y = preprocessed_data[target_col]

    print("Step 3: Training and evaluating the model...")
    pipeline = get_features_and_model_pipeline()
    model, odds_df = cross_validate(
        X, y, pipeline, odds_df, betting_logic.simple_percentage
    )
