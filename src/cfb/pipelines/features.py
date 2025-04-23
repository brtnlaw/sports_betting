import datetime as dt
from typing import List

import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

set_config(transform_output="pandas")


class DaysSinceLastGameTransformer(BaseEstimator, TransformerMixin):
    """Generates the days since last game for each team. If first game, fills since 1/1/2000."""

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates days since last game column.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        X_ = X.copy()
        df_home = X_[["start_date", "home_team"]].rename(columns={"home_team": "team"})
        df_away = X_[["start_date", "away_team"]].rename(columns={"away_team": "team"})
        df_combined = pd.concat([df_home, df_away]).sort_values(
            by=["team", "start_date"]
        )
        df_combined["previous_game"] = df_combined.groupby("team")["start_date"].shift(
            1
        )
        X_["previous_game"] = None
        for side in ["home", "away"]:
            # Maintain the original index through merge by resetting, and then setting again
            X_ = (
                X_.reset_index()
                .merge(
                    df_combined[["team", "start_date", "previous_game"]],
                    left_on=[f"{side}_team", "start_date"],
                    right_on=["team", "start_date"],
                    how="left",
                    suffixes=("", f"_{side}"),
                )
                .set_index(X_.index.name)
            )
            X_[f"previous_game_{side}"].fillna(dt.date(2000, 1, 1), inplace=True)
            X_[f"{side}_days_since_last_game"] = (
                X_["start_date"] - X_[f"previous_game_{side}"]
            ).apply(lambda x: x.days)
        X_.drop(
            columns=["team", "previous_game_home", "previous_game_away"],
            inplace=True,
            errors="ignore",
        )
        return X_


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
        Generates rolling column to DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
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


def offense_pipeline() -> Pipeline:
    """
    Pipeline for all offensive features.

    Returns:
        Pipeline: Pipeline with offensive features.
    """
    offense_pipeline = Pipeline(
        [
            (
                "rolling_offense_1_3_5",
                RollingTransformer(
                    "points_for", "home_points", "away_points", [1, 3, 5], 1, "mean"
                ),
            ),
            (
                "rolling_third_down_attempts_1_3_5",
                RollingTransformer(
                    "third_down_attempts",
                    "home_third_down_attempts",
                    "away_third_down_attempts",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_third_down_successes_1_3_5",
                RollingTransformer(
                    "third_down_successes",
                    "home_third_down_successes",
                    "away_third_down_successes",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_fourth_down_attempts_1_3_5",
                RollingTransformer(
                    "fourth_down_attempts",
                    "home_fourth_down_attempts",
                    "away_fourth_down_attempts",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_fourth_down_successes_1_3_5",
                RollingTransformer(
                    "fourth_down_successes",
                    "home_fourth_down_successes",
                    "away_fourth_down_successes",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_passing_yds_for_1_3_5",
                RollingTransformer(
                    "passing_yds_for",
                    "home_net_passing_yards",
                    "away_net_passing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_ints_thrown_1_3_5",
                RollingTransformer(
                    "ints_thrown",
                    "home_interceptions",
                    "away_interceptions",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_rushing_yds_for_1_3_5",
                RollingTransformer(
                    "rushing_yds_for",
                    "home_rushing_yards",
                    "away_rushing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_passing_tds_1_3_5",
                RollingTransformer(
                    "passing_tds",
                    "home_passing_tds",
                    "away_passing_tds",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_rushing_tds_1_3_5",
                RollingTransformer(
                    "rushing_tds",
                    "home_rushing_tds",
                    "away_rushing_tds",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
        ]
    )
    return offense_pipeline


def defense_pipeline() -> Pipeline:
    """
    Pipeline for all defensive features.

    Returns:
        Pipeline: Pipeline with defensive features.
    """
    defense_pipeline = Pipeline(
        [
            (
                "rolling_defense_1_3_5",
                RollingTransformer(
                    "points_against", "away_points", "home_points", [1, 3, 5], 1, "mean"
                ),
            ),
            (
                "rolling_passing_yds_given_up_1_3_5",
                RollingTransformer(
                    "passing_yds_given_up",
                    "away_net_passing_yards",
                    "home_net_passing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
        ]
    )
    return defense_pipeline


def feature_pipeline() -> Pipeline:
    """
    Combines all types of features into one pipeline.

    Returns:
        Pipeline: Combined pipeline of features.
    """
    pipeline = Pipeline(
        [
            ("days_since", DaysSinceLastGameTransformer()),
            ("offense_pipeline", offense_pipeline()),
            ("defense_pipeline", defense_pipeline()),
        ]
    )
    return pipeline
