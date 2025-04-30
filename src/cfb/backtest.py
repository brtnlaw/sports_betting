import argparse
import datetime as dt
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from data.data_prep import DataPrep
from pipelines.pipeline import get_features_and_model_pipeline
from pipelines.preprocessing import get_preprocess_pipeline
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from strategy.betting_logic import BettingLogic

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())


class RollingTimeSeriesSplit(BaseCrossValidator):
    """Custom rolling cross-validator that rolls over the training window by season, to avoid model drift."""

    def __init__(self, seasons: pd.Series, fixed_window_size: int = 5):
        """
        Initializes seasons Series and window size.

        Args:
            seasons (pd.Series): Seasons Series with corresponding index.
            fixed_window_size (int, optional): Number of seasons in rolling window. Defaults to 5.
        """
        self.seasons = seasons
        self.fixed_window_size = fixed_window_size
        self.unique_seasons = sorted(seasons.unique())

    def split(self, X, y=None, groups=None):
        """Yields indices for train-test splits."""
        for i in range(self.fixed_window_size, len(self.unique_seasons)):
            train_seasons = self.unique_seasons[i - self.fixed_window_size : i]
            test_season = self.unique_seasons[i]

            train_idx = np.where(self.seasons.isin(train_seasons))[0]
            test_idx = np.where(self.seasons == test_season)[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splits."""
        return len(self.unique_seasons) - self.fixed_window_size


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    odds_df: pd.DataFrame,
    betting_fnc: str = "spread_probs",
    fixed_window_size: int = 5,
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
        betting_fnc (str, optional): Function to determine bets. Defaults to "spread_probs".
        fixed_window_size (int, optional): Seasons to train on. Defaults to 5.
        file_name (str, optional): Desired file name for model. Defaults to None.

    Returns:
        tuple[Pipeline, pd.DataFrame]: The total pipeline that has been fit and the bets made.
    """
    cv_split = RollingTimeSeriesSplit(
        seasons=X["season"], fixed_window_size=fixed_window_size
    )
    contrib_df_list = []
    betting_logic = BettingLogic(betting_fnc)

    for train_idx, test_idx in cv_split.split(X, y):
        # Train-test split
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        # Get predictions
        preds = pipeline.predict(X_test, pred_contrib=True)
        cols = pipeline.named_steps["light_gbm"].feature_name_ + ["bias"]
        contrib_df_list.append(
            pd.DataFrame(preds[:, :], columns=cols, index=X_test.index)
        )
        odds_df.iloc[test_idx, odds_df.columns.get_loc("pred")] = preds.sum(axis=1)

    contrib_df = pd.concat(contrib_df_list).sort_index()
    odds_df = betting_logic.apply_bets(odds_df)

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
        contrib_df,
        os.path.join(PROJECT_ROOT, f"src/cfb/models/{file_name}_{y.name}_contrib.pkl"),
    )
    joblib.dump(
        odds_df,
        os.path.join(
            PROJECT_ROOT,
            f"src/cfb/models/{file_name}_{y.name}_{betting_fnc}.pkl",
        ),
    )
    return pipeline, odds_df


if __name__ == "__main__":
    """
    Example usage:
    python src/cfb/backtest.py
    python src/cfb/backtest.py --name "baseline"
    python src/cfb/backtest.py --name "baseline" --betting_fnc "spread_probs"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Model file name to save.")
    parser.add_argument("--betting_fnc", type=str, help="Betting function to apply.")
    args = parser.parse_args()

    print("Step 1: Loading data...")
    data_prep = DataPrep(dataset="cfb")
    raw_data = data_prep.get_data()

    print("Step 2: Preprocess and separate odds, X, and y...")
    preprocessed_data = get_preprocess_pipeline().fit_transform(raw_data)
    target_line_dict = {
        "total": ["min_ou", "max_ou"],
        "home_away_spread": ["min_spread", "max_spread"],
    }

    target_col = "home_away_spread"
    betting_cols = target_line_dict[target_col]
    all_betting_cols = [
        line for line_list in target_line_dict.values() for line in line_list
    ]

    odds_df = preprocessed_data[[target_col] + betting_cols]
    odds_df.loc[:, "pred"] = None
    X = preprocessed_data.drop(columns=[target_col] + all_betting_cols)
    y = preprocessed_data[target_col]

    print("Step 3: Training and evaluating the model...")
    pipeline = get_features_and_model_pipeline()
    cross_val_kwargs = {}
    if args.name is not None:
        cross_val_kwargs["file_name"] = args.name
    if args.betting_fnc is not None:
        cross_val_kwargs["betting_fnc"] = args.betting_fnc

    model, odds_df = cross_validate(X, y, pipeline, odds_df, **cross_val_kwargs)

    print("Success!")
