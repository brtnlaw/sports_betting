import argparse
import datetime as dt
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from data.data_prep import DataPrep
from flaml.tune import SearchCV
from pipelines.pipeline import get_features_and_model_pipeline
from pipelines.preprocessing import get_preprocess_pipeline
from scipy.stats import randint
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from strategy.betting_logic import BettingLogic

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())


class RollingTimeSeriesSplit(BaseCrossValidator):
    """
    Custom rolling cross-validator that rolls over the training window by season, to avoid model drift.
    Example: Train on 2018-2022, test on 2023, then train on 2019-2023, test on 2024, then train on 2020-2024, test on 2025.
    """

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
    train_window_size: int = 4,
    validation_window_size: int = 1,
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
        train_window_size (int, optional): Seasons to train on. Defaults to 5.
        validation_window_size (int, optional): Seasons to validate on. Defaults to 1.
        file_name (str, optional): Desired file name for model. Defaults to None.

    Returns:
        tuple[Pipeline, pd.DataFrame]: The total pipeline that has been fit and the bets made.
    """
    train_test_split = RollingTimeSeriesSplit(
        seasons=X["season"],
        fixed_window_size=train_window_size + validation_window_size,
    )
    contrib_df_list = []
    betting_logic = BettingLogic(betting_fnc)

    # Initial train split
    first_split = True
    odds_df["is_train"] = False

    # Hyperparameter distributions for LightGBM
    param_distributions = [
        {
            "light_gbm__learning_rate": [0.01, 0.05, 0.1],
            "light_gbm__num_leaves": randint(30, 100),
            "light_gbm__max_depth": randint(3, 10),
        },
    ]

    # Rolls the training window to accumulate years
    fold = 1
    for train_val_idx, test_idx in train_test_split.split(X, y):
        print(f"Beginning fold {fold}...")
        # (Train-validation)-test split
        X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
        X_test = X.iloc[test_idx]

        # Separate the train_val into training and validation to do hyperparameter search
        train_val_split = RollingTimeSeriesSplit(
            seasons=X_train_val["season"], fixed_window_size=train_window_size
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=1,
            cv=train_val_split,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        start = time.time()
        search.fit(X_train_val, y_train_val)
        cv_pipeline = search.best_estimator_
        end = time.time()
        print(f"Hyperparameter tuning took {end-start:.2f} seconds")

        # NOTE: Can consider storing validation error, but not necessary
        start = time.time()
        cv_pipeline.fit(X_train_val, y_train_val)
        end = time.time()
        print(f"Train data fitting took {end-start:.2f} seconds")

        # If this is the first split, we denote that it's a training prediction before saving df
        if first_split:
            odds_df.iloc[train_val_idx, odds_df.columns.get_loc("pred")] = (
                cv_pipeline.predict(X_train_val)
            )
            # is_train inclusive of validation, proxy for data points we will see
            # Used for the train-test metrics
            odds_df.iloc[train_val_idx, odds_df.columns.get_loc("is_train")] = True
            first_split = False

        # Get predictions to odds_df, appends feature contributions
        start = time.time()
        preds = cv_pipeline.predict(X_test, pred_contrib=True)
        end = time.time()
        print(f"Test predicting took {end-start:.2f} seconds")
        cols = cv_pipeline.named_steps["light_gbm"].feature_name_ + ["bias"]
        contrib_df_list.append(
            pd.DataFrame(preds[:, :], columns=cols, index=X_test.index)
        )
        odds_df.iloc[test_idx, odds_df.columns.get_loc("pred")] = preds.sum(axis=1)
        fold += 1

    contrib_df = pd.concat(contrib_df_list).sort_index()
    odds_df = betting_logic.apply_bets(odds_df)

    if not file_name:
        td = dt.datetime.today()
        file_name = f"model_{td.month}_{td.day}_{td.year%100}"

    # Save the pipeline and odds_df with the betting fnc name
    joblib.dump(
        cv_pipeline,
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
    start = time.time()

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

    end = time.time()
    print("Success!")
    print(f"Elapsed time: {end - start:.2f} seconds")
