import datetime as dt

import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

set_config(transform_output="pandas")


class DaysSinceLastGameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
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
    def __init__(
        self, new_col, home_col, away_col, window_size, min_periods=1, agg_func="mean"
    ):
        self.new_col = new_col
        self.home_col = home_col
        self.away_col = away_col
        self.window_size = window_size
        self.min_periods = min_periods
        self.agg_func = agg_func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
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

        func_map = {
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
            "max": lambda x: x.max(),
            "min": lambda x: x.min(),
        }
        rolling_col_name = f"rolling_{self.window_size}_{self.agg_func}_{self.new_col}"
        game_df[rolling_col_name] = (
            game_df.groupby("team")[self.new_col]
            .shift(1)
            .rolling(window=self.window_size, min_periods=self.min_periods)
            .apply(func_map[self.agg_func], raw=True)
            .fillna(0)
        )
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


def offense_pipeline():
    offense_pipeline = Pipeline(
        [
            (
                "rolling_offense_3",
                RollingTransformer(
                    "points_for", "home_points", "away_points", 3, 1, "mean"
                ),
            ),
            (
                "rolling_offense_5",
                RollingTransformer(
                    "points_for", "home_points", "away_points", 5, 1, "mean"
                ),
            ),
        ]
    )
    return offense_pipeline


def defense_pipeline():
    defense_pipeline = Pipeline(
        [
            (
                "rolling_defense_3",
                RollingTransformer(
                    "points_against", "away_points", "home_points", 3, 1, "mean"
                ),
            ),
            (
                "rolling_defense_5",
                RollingTransformer(
                    "points_against", "away_points", "home_points", 5, 1, "mean"
                ),
            ),
        ]
    )
    return defense_pipeline


def feature_pipeline():
    pipeline = Pipeline(
        [
            ("days_since", DaysSinceLastGameTransformer()),
            ("offense_pipeline", offense_pipeline()),
            ("defense_pipeline", defense_pipeline()),
        ]
    )
    return pipeline
