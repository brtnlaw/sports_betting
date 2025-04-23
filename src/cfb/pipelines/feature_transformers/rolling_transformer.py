from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RollingTransformer(BaseEstimator, TransformerMixin):
    """Helper to generate a rolling version of a given column. Handles both home and away teams at the same time."""

    def __init__(
        self,
        new_col: str,
        home_col: str,
        away_col: str,
        window_sizes: List[int],
        min_periods: int = 1,
        agg_func: str = "mean",
    ):
        """
        Initializes class to generate rolling column.

        Args:
            new_col (str): Name of the new column.
            home_col (str): Data representing home team to roll.
            away_col (str): Data representing away team to roll.
            window_size (int): Window sizes to roll.
            min_periods (int, optional): Minimum entries for a window if not enough data. Defaults to 1.
            agg_func (str, optional): Function that aggregates window data. Defaults to "mean".
        """
        self.new_col = new_col
        self.home_col = home_col
        self.away_col = away_col
        self.window_sizes = window_sizes
        self.min_periods = min_periods
        self.agg_func = agg_func

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates rolling column(s) to DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with rolled values.
        """
        X_ = X.copy()

        # Separately treats home and away, combines into game_df
        home_df = X_[
            [
                "start_date",
                "home_team",
                self.home_col,
            ]
        ].rename(
            columns={
                "home_team": "team",
                self.home_col: self.new_col,
            }
        )
        away_df = X_[["start_date", "away_team", self.away_col]].rename(
            columns={
                "away_team": "team",
                self.away_col: self.new_col,
            }
        )
        game_df = pd.concat([home_df, away_df])
        game_df.sort_values(by=["team", "start_date"], inplace=True)
        game_df.drop_duplicates(
            subset=["start_date", "team"], keep="last", inplace=True
        )

        # Create game_df rolling_column according to specs
        func_map = {
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
            "max": lambda x: x.max(),
            "min": lambda x: x.min(),
        }

        for window_size in self.window_sizes:
            rolling_col_name = f"rolling_{window_size}_{self.agg_func}_{self.new_col}"
            game_df[rolling_col_name] = (
                game_df.groupby("team")[self.new_col]
                .shift(1)
                .rolling(window=window_size, min_periods=self.min_periods)
                .apply(func_map[self.agg_func], raw=True)
                .fillna(0)
            )
            # Merge X_ with game_df, starting with home then away, retaining original index
            X_ = (
                X_.reset_index()
                .merge(
                    game_df[["start_date", "team", rolling_col_name]],
                    left_on=["start_date", "home_team"],
                    right_on=["start_date", "team"],
                    how="left",
                )
                .set_index(X_.index.name)
            )
            X_ = X_.rename(columns={rolling_col_name: f"home_{rolling_col_name}"}).drop(
                columns=["team"]
            )
            X_ = (
                X_.reset_index()
                .merge(
                    game_df[["start_date", "team", rolling_col_name]],
                    left_on=["start_date", "away_team"],
                    right_on=["start_date", "team"],
                    how="left",
                )
                .set_index(X_.index.name)
            )
            X_ = X_.rename(columns={rolling_col_name: f"away_{rolling_col_name}"}).drop(
                columns=["team"]
            )
        return X_
